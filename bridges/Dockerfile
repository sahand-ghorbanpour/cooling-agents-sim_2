# ========== bridges/Dockerfile (for NATS bridge and scheduler) ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="Bridge Services for LangGraph Integration"

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
COPY bridges/ ./bridges/
COPY graphs/ ./graphs/
COPY common/ ./common/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd --create-home --shell /bin/bash bridges
RUN chown -R bridges:bridges /app
USER bridges

# Health check  
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import bridges.nats_bridge; print('OK')" || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "bridges/nats_bridge.py"]
