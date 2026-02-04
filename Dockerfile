# WhisperX CPU-only Dockerfile for EasyPanel
# Versão simplificada sem diarização (pyannote-audio é muito pesado)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU-only (versão menor e mais estável)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    python-multipart==0.0.6

# Install data processing
RUN pip install --no-cache-dir \
    numpy==1.26.3 \
    pandas==2.1.4

# Install faster-whisper (core transcription)
RUN pip install --no-cache-dir \
    faster-whisper==0.10.0 \
    ctranslate2==4.0.0

# Install transformers for alignment
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    huggingface-hub==0.20.2

# Install nltk
RUN pip install --no-cache-dir nltk==3.8.1

# Copy API file
COPY api.py .

# Create directories
RUN mkdir -p /app/uploads /app/outputs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
