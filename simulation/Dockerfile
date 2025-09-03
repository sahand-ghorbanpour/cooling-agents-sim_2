# ========== simulation/Dockerfile (for FMU environment) ==========
FROM continuumio/miniconda3:latest

LABEL maintainer="Cooling System Team"
LABEL description="Physics Simulation Environment with FMU Support"

WORKDIR /app

# Install system dependencies for FMU
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install conda packages for FMU support
RUN conda install -c conda-forge pyfmi=2.16.2 numpy pandas -y

# Install remaining packages via pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulation environment
COPY env/ ./env/
COPY simulation/ ./simulation/
COPY common/ ./common/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create simulation data directory
RUN mkdir -p /app/simulation_data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from env.frontier_env import SmallFrontierModel; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "from env.frontier_env import SmallFrontierModel; env = SmallFrontierModel(); print('Simulation environment ready')"]
