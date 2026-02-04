# WhisperX CPU-only Dockerfile for EasyPanel
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies (CPU-only)
RUN pip install --no-cache-dir \
    whisperx \
    fastapi \
    uvicorn \
    python-multipart

# Create directory for uploads
RUN mkdir -p /app/uploads /app/outputs

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
