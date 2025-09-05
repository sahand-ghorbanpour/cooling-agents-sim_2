# ========== services/control_runtime/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="Real-time Control Runtime Service"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app DEBIAN_FRONTEND=noninteractive

# System deps (curl for healthcheck; add others as needed)
RUN apt-get update && apt-get install -y \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy service-specific requirements and install
# (requirements.txt MUST live next to this Dockerfile)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy ONLY the control-runtime code inside this service folder
# Because build context = services/control_runtime, "." == that folder
COPY . /app/services/control_runtime/

# (We DO NOT copy ../common here; it's mounted read-only via a volume)

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash control \
 && mkdir -p /app/config /app/data /app/logs \
 && chown -R control:control /app
USER control

# Health check (requires curl installed above)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:9010/healthz || exit 1

EXPOSE 9010
CMD ["python", "services/control_runtime/main.py"]
