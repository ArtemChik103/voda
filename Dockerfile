# Dockerfile for KosmoHackathon 2026 - Operational Hydrological Monitoring
# Case: Sentinel-1 and Sentinel-2 Multi-modal Flood Detection
FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system GIS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python scientific and geospatial dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

# Default entrypoint runs test suite and metric validation
CMD ["python", "-m", "pytest", "tests/", "-v"]
