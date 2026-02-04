"""
WhisperX API Server for EasyPanel deployment (CPU-only)
Full features: Transcription + Alignment + Diarization
"""
import os
import tempfile
import uuid
import gc
from typing import Optional

import whisperx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

app = FastAPI(
    title="WhisperX API",
    description="Speech Recognition with Word-level Timestamps and Speaker Diarization",
    version="1.0.0"
)

# Configuration for CPU-only mode
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 4
DEFAULT_MODEL = "base"

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
        "compute_type": COMPUTE_TYPE,
        "features": ["transcription", "alignment", "diarization"]
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
    Transcribe an audio file with optional alignment and diarization
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.)
    - model: Whisper model size (tiny, base, small, medium, large-v2)
    - language: Language code (e.g., 'en', 'pt', 'es'). Auto-detect if not provided.
    - align: Enable word-level alignment (default: True)
    - diarize: Enable speaker diarization (requires hf_token)
    - hf_token: HuggingFace token for diarization
    - min_speakers: Minimum number of speakers
    - max_speakers: Maximum number of speakers
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if diarize and not hf_token:
        raise HTTPException(
            status_code=400, 
            detail="hf_token is required for diarization. Get one at https://huggingface.co/settings/tokens"
        )
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load model and audio
        whisper_model = get_model(model)
        audio = whisperx.load_audio(temp_path)
        
        # 1. Transcribe
        result = whisper_model.transcribe(
            audio, 
            batch_size=BATCH_SIZE,
            language=language
        )
        
        detected_language = result.get("language", language)
        
        # 2. Align (word-level timestamps)
        if align and result.get("segments"):
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language, 
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
                del model_a
                gc.collect()
            except Exception as e:
                print(f"Alignment failed: {e}")
        
        # 3. Diarize (speaker identification)
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
                del diarize_model
                gc.collect()
            except Exception as e:
                print(f"Diarization failed: {e}")
                raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")
        
        return JSONResponse(content={
            "success": True,
            "language": detected_language,
            "segments": result.get("segments", []),
            "word_segments": result.get("word_segments", [])
        })
        
    except HTTPException:
        raise
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
            {"name": "tiny", "params": "39M", "speed": "fastest"},
            {"name": "base", "params": "74M", "speed": "fast"},
            {"name": "small", "params": "244M", "speed": "medium"},
            {"name": "medium", "params": "769M", "speed": "slow"},
            {"name": "large-v2", "params": "1550M", "speed": "slowest"},
        ],
        "recommended_for_cpu": "base or small"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
