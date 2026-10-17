# Build & run on IBM LinuxONE (s390x) with Podman:
#   podman build -t sepsis-triage-api .
#   podman run -d -p 8050:8050 --name sepsis-api sepsis-triage-api
#   curl localhost:8050/health
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ make libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py model_meta.json tier1_lgb.txt tier2_lgb.txt ae_encoder.npz ./
EXPOSE 8050
USER 1001
CMD ["python", "app.py"]
