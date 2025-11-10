# api/Dockerfile  (we will build it from the REPO ROOT using -f)
FROM mcr.microsoft.com/devcontainers/python:3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (from api/requirements.txt)
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the whole api package (now includes __init__.py)
COPY api /app/api

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
