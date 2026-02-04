# WhisperX CPU-only Dockerfile for EasyPanel
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch CPU-only first (smaller download, more stable)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies in smaller batches
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    numpy \
    pandas

# Install whisper dependencies
RUN pip install --no-cache-dir \
    faster-whisper \
    ctranslate2

# Install alignment dependencies  
RUN pip install --no-cache-dir \
    transformers \
    nltk

# Install whisperx (without deps since we installed them)
RUN pip install --no-cache-dir whisperx --no-deps || \
    pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git --no-deps

# Install remaining whisperx dependencies
RUN pip install --no-cache-dir \
    pyannote-audio \
    huggingface-hub

# Copy project files
COPY api.py .

# Create directory for uploads
RUN mkdir -p /app/uploads /app/outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
