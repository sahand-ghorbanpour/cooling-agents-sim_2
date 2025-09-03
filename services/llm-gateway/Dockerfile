# ========== services/llm-gateway/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="LLM Gateway with NIM/OpenAI Proxy"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/llm-gateway/ ./services/llm-gateway/
COPY common/ ./common/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9000/healthz || exit 1

# Expose port
EXPOSE 9000

# Run the application
CMD ["uvicorn", "services.llm_gateway.main:app", "--host", "0.0.0.0", "--port", "9000"]