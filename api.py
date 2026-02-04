"""
WhisperX API Server for EasyPanel deployment (CPU-only)
"""
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import whisperx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

app = FastAPI(
    title="WhisperX API",
    description="Automatic Speech Recognition with Word-level Timestamps",
    version="1.0.0"
)

# Configuration for CPU-only mode
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 4
DEFAULT_MODEL = "base"  # Use smaller model for CPU: tiny, base, small, medium, large-v2

# Cache for loaded models
model_cache = {}


def get_model(model_name: str = DEFAULT_MODEL):
    """Load and cache the whisper model"""
    if model_name not in model_cache:
        print(f"Loading model: {model_name}")
        model_cache[model_name] = whisperx.load_model(
            model_name, 
            DEVICE, 
            compute_type=COMPUTE_TYPE
        )
    return model_cache[model_name]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "WhisperX API",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE
    }


@app.get("/health")
async def health():
    """Health check for EasyPanel"""
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL),
    language: Optional[str] = Form(default=None),
    align: bool = Form(default=True),
    diarize: bool = Form(default=False),
    hf_token: Optional[str] = Form(default=None),
    min_speakers: Optional[int] = Form(default=None),
    max_speakers: Optional[int] = Form(default=None),
):
    """
    Transcribe an audio file
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.)
    - model: Whisper model size (tiny, base, small, medium, large-v2)
    - language: Language code (e.g., 'en', 'pt', 'es'). Auto-detect if not provided.
    - align: Enable word-level alignment (default: True)
    - diarize: Enable speaker diarization (requires hf_token)
    - hf_token: HuggingFace token for diarization
    - min_speakers: Minimum number of speakers (for diarization)
    - max_speakers: Maximum number of speakers (for diarization)
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # Write file to disk
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load model
        whisper_model = get_model(model)
        
        # Load audio
        audio = whisperx.load_audio(temp_path)
        
        # Transcribe
        result = whisper_model.transcribe(
            audio, 
            batch_size=BATCH_SIZE,
            language=language
        )
        
        # Align (word-level timestamps)
        if align and result.get("segments"):
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=DEVICE
                )
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    DEVICE, 
                    return_char_alignments=False
                )
            except Exception as e:
                print(f"Alignment failed: {e}")
                # Continue without alignment
        
        # Diarize (speaker identification)
        if diarize and hf_token:
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token, 
                    device=DEVICE
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                print(f"Diarization failed: {e}")
                # Continue without diarization
        
        return JSONResponse(content={
            "success": True,
            "language": result.get("language"),
            "segments": result.get("segments", []),
            "word_segments": result.get("word_segments", [])
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.get("/models")
async def list_models():
    """List available Whisper models"""
    return {
        "models": [
            {"name": "tiny", "description": "Fastest, lowest accuracy", "vram": "~1GB"},
            {"name": "base", "description": "Fast, good for simple audio", "vram": "~1GB"},
            {"name": "small", "description": "Balanced speed/accuracy", "vram": "~2GB"},
            {"name": "medium", "description": "Good accuracy", "vram": "~5GB"},
            {"name": "large-v2", "description": "Best accuracy, slowest", "vram": "~10GB"},
            {"name": "large-v3", "description": "Latest, best accuracy", "vram": "~10GB"},
        ],
        "recommended_for_cpu": "base or small"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
