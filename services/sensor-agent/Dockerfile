# services/sensor-agent/Dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    build-essential gfortran curl \
 && rm -rf /var/lib/apt/lists/*

 RUN conda install -c conda-forge pyfmi=2.16.2 numpy pandas -y

# Copy only what we need early to leverage caching
# (adjust these paths to match where you placed the files)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# If you have python package code to copy (keep this narrow for caching)
COPY . /app/services/sensor-agent/
RUN mkdir -p /app/simulation_data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 9012

# Healthcheck is fine as you had it, or omit during build-debug:
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#   CMD python -c "from env.frontier_env import SmallFrontierModel; print('FMU OK')" || exit 1
CMD ["python", "services/sensor-agent/main.py"]
