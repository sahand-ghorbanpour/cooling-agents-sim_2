# ========== services/sensor-gateway/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="Sensor Gateway for PLC/Modbus Integration"

WORKDIR /app

# Install system dependencies for sensor protocols
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libmodbus-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/sensor-gateway/ ./services/sensor-gateway/
COPY common/ ./common/

# Create directories for sensor configurations and data
RUN mkdir -p /app/config /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9011/healthz || exit 1

# Expose port
EXPOSE 9011

# Run the application
CMD ["python", "services/sensor_gateway/main.py"]