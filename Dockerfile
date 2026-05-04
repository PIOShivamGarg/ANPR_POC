# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Copy the model file (ensure this exists in your project root)
COPY license_plate_detector.pt .

# Copy all images from local folder into container
COPY ["Vehicle License Plate List", "/app/Vehicle License Plate List"]

# Set environment variables to avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]