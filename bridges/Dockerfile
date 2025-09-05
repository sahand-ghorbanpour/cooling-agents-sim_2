# ========== bridges/Dockerfile ==========
FROM python:3.11-slim

LABEL maintainer="Cooling System Team"
LABEL description="Bridge Services for LangGraph Integration"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app DEBIAN_FRONTEND=noninteractive

# System deps (curl only used if you add an HTTP healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps (requirements.txt must be inside ./bridges)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy ONLY the bridges code from the build context
# (context = ./bridges, so "." == bridges folder)
COPY . /app/bridges/

# We DO NOT copy ../graphs or ../common here; they are mounted read-only via volumes

# Non-root user (optional but recommended)
RUN useradd --create-home --shell /bin/bash bridges \
 && chown -R bridges:bridges /app
USER bridges

# Lightweight healthcheck: import succeeds → OK
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import bridges.nats_bridge as _; print('OK')" || exit 1

# Default command (compose overrides for nats-bridge; scheduler can reuse this image)
CMD ["python", "bridges/nats_bridge.py"]
