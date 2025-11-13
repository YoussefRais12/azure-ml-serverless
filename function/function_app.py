# function/function_app.py
import os, io, csv, json, logging, datetime as dt
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = func.FunctionApp()

API_URL = os.environ["API_URL"]  # e.g. https://.../predict
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))   # tune in App Settings
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "120")) # tune in App Settings
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", "output")

session = requests.Session()
retries = Retry(total=4, backoff_factor=0.7, status_forcelist=(429,500,502,503,504))
session.mount("https://", HTTPAdapter(max_retries=retries))

@app.function_name(name="process_blob")
@app.blob_trigger(arg_name="inputblob",
                  path="input/{name}",
                  connection="AzureWebJobsStorage")
def process_blob(inputblob: func.InputStream):
    logging.info(f"[CFG] API_URL={API_URL} CHUNK_SIZE={CHUNK_SIZE} TIMEOUT_SEC={TIMEOUT_SEC}")
    conn_str = os.environ["AzureWebJobsStorage"]
    bsc = BlobServiceClient.from_connection_string(conn_str)
    out_container = bsc.get_container_client(OUTPUT_CONTAINER)

    # Read blob content (UTF-8; tolerate BOM)
    raw = inputblob.read()
    text = raw.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    # Detect delimiter (fallback to comma)
    first_line = text.splitlines(True)[0] if text else ""
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=[",",";","\t","|"])
        delim = dialect.delimiter
    except Exception:
        delim = ","
    logging.info(f"[INFO] Using delimiter: {repr(delim)}")

    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    if not reader.fieldnames:
        _write_meta(out_container, inputblob.name, {"error":"no_header"})
        return

    # Normalize headers
    headers = [h.strip().lower() for h in reader.fieldnames]
    id_key = "id" if "id" in headers else ("tweet_id" if "tweet_id" in headers else None)
    text_key = "text" if "text" in headers else ("tweet" if "tweet" in headers else None)
    logging.info(f"[INFO] headers={headers}, id_key={id_key}, text_key={text_key}")

    rows = []
    for r in reader:
        _id = (r.get(id_key) if id_key else None) or str(len(rows)+1)
        _tx = r.get(text_key) or r.get("comment") or r.get("message") or ""
        _tx = " ".join((_tx or "").split())
        if _tx:
            rows.append({"id": str(_id), "text": _tx})

    if not rows:
        _write_meta(out_container, inputblob.name, {"input_rows":0,"predictions":[],"note":"no_valid_rows"})
        return

    logging.info(f"[INFO] total_rows={len(rows)} — starting chunked inference")
    chunk_idx = 0
    total_preds = 0
    for i in range(0, len(rows), CHUNK_SIZE):
        batch = rows[i:i+CHUNK_SIZE]
        payload = {"items": batch}
        try:
            resp = session.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
            resp.raise_for_status()
            preds = resp.json()
        except Exception as e:
            # write an error file for this chunk so you see progress & cause
            err = {"error": f"api_call_failed: {str(e)}", "chunk_index": chunk_idx, "at": _now()}
            _upload_json(out_container, _chunk_blob_name(inputblob.name, chunk_idx), err)
            logging.error(f"[ERR] chunk {chunk_idx} failed: {e}")
            chunk_idx += 1
            continue

        # Write this chunk’s predictions immediately
        out_obj = {
            "chunk_index": chunk_idx,
            "chunk_size": len(batch),
            "predictions": preds,
            "processed_at": _now()
        }
        _upload_json(out_container, _chunk_blob_name(inputblob.name, chunk_idx), out_obj)
        total_preds += len(preds)
        logging.info(f"[OK] wrote chunk {chunk_idx} with {len(preds)} predictions")
        chunk_idx += 1

    # Write a small manifest summary
    manifest = {
        "file": inputblob.name,
        "total_rows": len(rows),
        "total_chunks": chunk_idx,
        "total_predictions": total_preds,
        "completed_at": _now()
    }
    _upload_json(out_container, _manifest_blob_name(inputblob.name), manifest)
    logging.info(f"[DONE] manifest written: {manifest}")

def _upload_json(container_client, blob_name, obj):
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    container_client.upload_blob(name=blob_name, data=data, overwrite=True)

def _chunk_blob_name(input_path_name, chunk_idx):
    # input_path_name like 'input/myfile.csv' — keep base name for output
    base = input_path_name.split("/")[-1]
    return f"{base}.{chunk_idx:04d}.json"   # e.g. myfile.csv.0000.json

def _manifest_blob_name(input_path_name):
    base = input_path_name.split("/")[-1]
    return f"{base}.manifest.json"

def _write_meta(container_client, input_path_name, obj):
    _upload_json(container_client, _manifest_blob_name(input_path_name), obj)

def _now():
    return dt.datetime.utcnow().isoformat() + "Z"
