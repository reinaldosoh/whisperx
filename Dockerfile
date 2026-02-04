# WhisperX com Diarização - CPU-only para EasyPanel
# Usando imagem base com PyTorch já instalado
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Forçar CPU (ignorar CUDA)
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_DEVICE="cpu"

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

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    python-multipart==0.0.6

# Install whisperx with all dependencies (including pyannote for diarization)
# Using --default-timeout to handle slow connections
RUN pip install --no-cache-dir --default-timeout=300 whisperx

# Copy API file
COPY api.py .

# Create directories
RUN mkdir -p /app/uploads /app/outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
