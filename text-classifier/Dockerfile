FROM python:3.11-slim

# Setze Arbeitsverzeichnis
WORKDIR /app

# Installiere System-Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiere Requirements
COPY requirements.txt .

# Installiere Python-Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere Service-Code
COPY classifier_service.py .

# Expose Port (nur im Docker-Netzwerk!)
EXPOSE 8000

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Starte Service
CMD ["uvicorn", "classifier_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]