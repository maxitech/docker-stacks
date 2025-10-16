FROM python:3.11-slim

# Setze Arbeitsverzeichnis
WORKDIR /app

# Installiere System-Dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*


# Kopiere Requirements
COPY requirements.txt .

# Installiere Python-Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere Service-Code
COPY classifier_service.py .

# Expose Port (nur im Docker-Netzwerk!)
EXPOSE 8000

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Starte Service
CMD ["uvicorn", "classifier_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]