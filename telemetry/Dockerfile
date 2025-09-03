FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="Telemetry Writer Service for InfluxDB"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY telemetry/ ./telemetry/
COPY common/ ./common/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd --create-home --shell /bin/bash telemetry
RUN chown -R telemetry:telemetry /app
USER telemetry

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import telemetry.influx_writer; print('OK')" || exit 1

# Run the application
CMD ["python", "telemetry/influx_writer.py"]