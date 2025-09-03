# ========== services/mcp-math/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="MCP Math Service for Thermal Optimization"

WORKDIR /app

# Install system dependencies for numerical computing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/mcp-math/ ./services/mcp-math/
COPY common/ ./common/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7000/healthz || exit 1

# Expose port
EXPOSE 7000

# Run the application
CMD ["uvicorn", "services.mcp_math.main:app", "--host", "0.0.0.0", "--port", "7000"]