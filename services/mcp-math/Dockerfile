# ========== services/mcp-math/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="MCP Math Service for Thermal Optimization"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app DEBIAN_FRONTEND=noninteractive

# System deps for numerical libs & healthcheck
RUN apt-get update && apt-get install -y \
    gcc g++ gfortran \
    libopenblas-dev liblapack-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps (requirements.txt must be in this folder)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy ONLY this service's code (context = services/mcp-math)
COPY . /app/services/mcp-math/
# Do NOT copy ../common here — it's mounted read-only via a volume

# Data/logs dirs if you need them
RUN mkdir -p /app/data /app/logs

# Health check (curl installed above)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:7000/healthz || exit 1

EXPOSE 7000
CMD ["uvicorn", "services.mcp-math.main:app", "--host", "0.0.0.0", "--port", "7000"]
