FROM python:3.10-slim

# Install system dependencies for OpenCV and OpenVINO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Upgrade pip and install requirements with a high timeout to prevent corruption on slow connections
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV CONFIG_PATH=config.docker.yaml

# Expose the API port
EXPOSE 8000

# Start the API server
CMD ["python", "api.py"]
