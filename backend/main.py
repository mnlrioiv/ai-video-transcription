import os
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
import ffmpeg
import json
import urllib.request

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
    model = genai.GenerativeModel('gemini-2.5-flash')
    # Prompt for short summary
    short_prompt = f"""Genera un resumen conciso (máximo 3-4 oraciones) del siguiente texto:
{text}
"""
    # Prompt for detailed summary
    detailed_prompt = f"""Genera un resumen detallado que capture los puntos principales, ejemplos y conclusiones del siguiente texto:
{text}
"""
    try:
        import time
        short_response = model.generate_content(short_prompt)
        time.sleep(2)  # Pequeña espera para evitar el error 429 (Too Many Requests)
        detailed_response = model.generate_content(detailed_prompt)
        return {
            "short_summary": short_response.text,
            "detailed_summary": detailed_response.text
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise e

def summarize_with_local_server(text: str):
    """Generate summaries using local LM Studio server (Gemma) as a fallback."""
    url = os.getenv("LOCAL_AI_URL", "http://localhost:1234/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    
    short_prompt = f"Genera un resumen conciso (máximo 3-4 oraciones) del siguiente texto:\n{text}"
    detailed_prompt = f"Genera un resumen detallado que capture los puntos principales, ejemplos y conclusiones del siguiente texto:\n{text}"
    
    def fetch_from_local(prompt):
        data = {
            "model": "gemma",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling local server: {e}")
            raise e

    try:
        short_summary = fetch_from_local(short_prompt)
        detailed_summary = fetch_from_local(detailed_prompt)
        return {
            "short_summary": short_summary,
            "detailed_summary": detailed_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ambas APIs (Gemini y Local) fallaron. Error: {e}")

import uuid
from fastapi import BackgroundTasks

# Store tasks in memory (task_id: {status, progress, result, error})
tasks = {}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

async def process_video_task(task_id: str, video_bytes: bytes, filename: str):
    tasks[task_id] = {"status": "processing", "progress": 0, "message": "Iniciando..."}
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, filename)
        audio_path = os.path.join(tmpdir, "audio.wav")
        
        try:
            # Save uploaded video
            tasks[task_id]["message"] = "Guardando archivo..."
            with open(video_path, "wb") as buffer:
                buffer.write(video_bytes)
            tasks[task_id]["progress"] = 10
            
            # Extract audio
            tasks[task_id]["message"] = "Extrayendo audio..."
            extract_audio(video_path, audio_path)
            tasks[task_id]["progress"] = 30
            
            # Transcribe
            tasks[task_id]["message"] = "Transcribiendo (esto puede tardar)..."
            # Note: Whisper doesn't easily give progress, so we jump to 80% after it finishes
            transcription = transcribe_audio(audio_path)
            tasks[task_id]["progress"] = 80
            
            # Summarize
            tasks[task_id]["message"] = "Generando resúmenes..."
            try:
                summaries = summarize_with_gemini(transcription)
            except Exception as e:
                logger.warning(f"Gemini falló: {e}. Intentando con servidor local (LM Studio)...")
                tasks[task_id]["message"] = "Gemini falló, intentando con servidor local..."
                summaries = summarize_with_local_server(transcription)
            tasks[task_id]["progress"] = 100
            
            # Finalize
            tasks[task_id] = {
                "status": "completed",
                "progress": 100,
                "result": {
                    "filename": filename,
                    "transcription": transcription,
                    "short_summary": summaries["short_summary"],
                    "detailed_summary": summaries["detailed_summary"]
                }
            }
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}")
            logger.error(traceback.format_exc())
            tasks[task_id] = {
                "status": "failed",
                "progress": 0,
                "error": str(e)
            }

@app.post("/transcribe")
async def transcribe_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "progress": 0}
    
    # Read file content immediately to avoid closing issues in background
    try:
        video_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {e}")
        
    background_tasks.add_task(process_video_task, task_id, video_bytes, file.filename)
    
    return {"task_id": task_id}

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