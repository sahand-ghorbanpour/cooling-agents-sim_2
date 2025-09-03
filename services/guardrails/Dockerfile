# ========== services/guardrails/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"  
LABEL description="Guardrails Service with Policy Enforcement"

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
COPY services/guardrails/ ./services/guardrails/
COPY common/ ./common/

# Set environment variables  
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "services.guardrails.main:app", "--host", "0.0.0.0", "--port", "8001"]