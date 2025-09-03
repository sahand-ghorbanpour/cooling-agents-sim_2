# services/sensor-agent/Dockerfile
FROM continuumio/miniconda3:latest

LABEL maintainer="Cooling System Team"
LABEL description="Physics Simulation Service with FMU Support"

WORKDIR /app

# Install system dependencies for FMU
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install conda packages for FMU support (CRITICAL!)
RUN conda install -c conda-forge pyfmi=2.16.2 numpy pandas -y

# Install remaining packages via pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulation code and environment
COPY services/sensor-agent/ ./services/sensor-agent/
COPY env/ ./env/  # Your FMU files
COPY common/ ./common/

# Create simulation data directory
RUN mkdir -p /app/simulation_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check using your actual environment
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from env.frontier_env import SmallFrontierModel; env = SmallFrontierModel(); print('FMU OK')" || exit 1

# Expose metrics port
EXPOSE 9012

# Run the sensor agent service
CMD ["python", "services/sensor-agent/main.py"]