import os
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
import ffmpeg

import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=api_key)

# Load Whisper model (choose base, small, medium, large)
# For MVP we can use "base" for speed/accuracy tradeoff
model = whisper.load_model("base")

app = FastAPI(title="Video Transcription & Summarization API")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"message": "Error interno del servidor", "detail": str(exc)}
    )

def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video file to wav format using ffmpeg."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e}")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper model."""
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_with_gemini(text: str):
    """Generate short and detailed summaries using Gemini."""
    model = genai.GenerativeModel('gemini-pro')
    # Prompt for short summary
    short_prompt = f"""Genera un resumen conciso (máximo 3-4 oraciones) del siguiente texto:
{text}
"""
    # Prompt for detailed summary
    detailed_prompt = f"""Genera un resumen detallado que capture los puntos principales, ejemplos y conclusiones del siguiente texto:
{text}
"""
    try:
        short_response = model.generate_content(short_prompt)
        detailed_response = model.generate_content(detailed_prompt)
        return {
            "short_summary": short_response.text,
            "detailed_summary": detailed_response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    # Validate file type (optional)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, file.filename)
        audio_path = os.path.join(tmpdir, "audio.wav")
        
        # Save uploaded video
        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception:
            raise HTTPException(status_code=500, detail="Could not save uploaded file")
        
        # Extract audio
        try:
            extract_audio(video_path, audio_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
        
        # Transcribe
        try:
            transcription = transcribe_audio(audio_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
        
        # Summarize
        try:
            summaries = summarize_with_gemini(transcription)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
        
        # Return results
        return JSONResponse(content={
            "filename": file.filename,
            "transcription": transcription,
            "short_summary": summaries["short_summary"],
            "detailed_summary": summaries["detailed_summary"]
        })

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}