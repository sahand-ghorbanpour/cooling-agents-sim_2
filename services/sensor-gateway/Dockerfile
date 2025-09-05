# ========== services/sensor-gateway/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="Sensor Gateway for PLC/Modbus Integration"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl libmodbus-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only the gateway code (not the whole repo)
COPY . /app/services/sensor-gateway/

# Create app dirs
RUN mkdir -p /app/config /app/data /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9011/healthz || exit 1

EXPOSE 9011
CMD ["python", "services/sensor-gateway/main.py"]
