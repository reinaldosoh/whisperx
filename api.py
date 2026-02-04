"""
Whisper API Server for EasyPanel deployment (CPU-only)
Uses faster-whisper for efficient transcription
"""
import os
import tempfile
import uuid
from typing import Optional

from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Whisper Transcription API",
    description="Fast Speech Recognition API using faster-whisper (CPU-only)",
    version="1.0.0"
)

# Configuration for CPU-only mode
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
DEFAULT_MODEL = "base"

# Cache for loaded models
model_cache = {}


def get_model(model_name: str = DEFAULT_MODEL):
    """Load and cache the whisper model"""
    if model_name not in model_cache:
        print(f"Loading model: {model_name}")
        model_cache[model_name] = WhisperModel(
            model_name, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE
        )
    return model_cache[model_name]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Whisper Transcription API",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "default_model": DEFAULT_MODEL
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
    task: str = Form(default="transcribe"),
):
    """
    Transcribe an audio file
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.)
    - model: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
    - language: Language code (e.g., 'en', 'pt', 'es'). Auto-detect if not provided.
    - task: 'transcribe' or 'translate' (translate to English)
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        whisper_model = get_model(model)
        
        segments, info = whisper_model.transcribe(
            temp_path,
            language=language,
            task=task,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )
        
        result_segments = []
        full_text = ""
        
        for segment in segments:
            seg_data = {
                "id": segment.id,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }
            
            if segment.words:
                seg_data["words"] = [
                    {
                        "word": word.word,
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                        "probability": round(word.probability, 3)
                    }
                    for word in segment.words
                ]
            
            result_segments.append(seg_data)
            full_text += segment.text
        
        return JSONResponse(content={
            "success": True,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "text": full_text.strip(),
            "segments": result_segments,
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.get("/models")
async def list_models():
    """List available Whisper models"""
    return {
        "models": [
            {"name": "tiny", "params": "39M", "description": "Fastest, lowest accuracy"},
            {"name": "base", "params": "74M", "description": "Fast, good for simple audio"},
            {"name": "small", "params": "244M", "description": "Balanced speed/accuracy"},
            {"name": "medium", "params": "769M", "description": "Good accuracy"},
            {"name": "large-v2", "params": "1550M", "description": "Best accuracy"},
            {"name": "large-v3", "params": "1550M", "description": "Latest, best accuracy"},
        ],
        "recommended_for_cpu": "base or small",
        "note": "Models are downloaded on first use"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
