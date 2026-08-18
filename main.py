import os
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr-worker")

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
MODEL_CACHE = os.environ.get("MODEL_CACHE", "/app/model_cache")
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from faster_whisper import WhisperModel
    logger.info(f"Loading model {MODEL_SIZE}...")
    t0 = time.time()
    # Используем INT8 для экономии памяти
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        download_root=MODEL_CACHE,
        cpu_threads=4,
        num_workers=2
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")
    yield
    model = None

app = FastAPI(title="ASR Worker", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SIZE}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "ru"):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    audio_bytes = await file.read()
    logger.info(f"Received {file.filename}: {len(audio_bytes)} bytes, lang={language}")
    return await _transcribe(audio_bytes, "audio.mp3", language)

@app.post("/transcribe-raw")
async def transcribe_raw(request: Request):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    language = request.headers.get("X-Language", "ru")
    audio_bytes = await request.body()
    logger.info(f"Received raw audio: {len(audio_bytes)} bytes, lang={language}")
    return await _transcribe(audio_bytes, "audio.mp3", language)

@app.post("/transcribe-json")
async def transcribe_json(request: Request):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    data = await request.json()
    audio_b64 = data.get("audio", "")
    lang = data.get("language", "ru")
    fmt = data.get("format", "mp3")
    audio_bytes = __import__("base64").b64decode(audio_b64)
    logger.info(f"Received JSON audio: {len(audio_bytes)} bytes, lang={lang}")
    return await _transcribe(audio_bytes, f"audio.{fmt}", lang)

async def _transcribe(audio_bytes, filename, language):
    tmp_path = f"/tmp/{filename}"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    
    t0 = time.time()
    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text)
        
        text = " ".join(text_parts)
        duration = time.time() - t0
        
        logger.info(f"Transcribed in {duration:.1f}s: {len(text)} chars")
        
        return {
            "text": text,
            "duration_sec": round(duration, 1),
            "language": info.language,
            "audio_duration_sec": round(info.duration, 1) if info.duration else 0
        }
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)