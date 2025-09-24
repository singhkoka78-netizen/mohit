import os
import uuid
import tempfile
import subprocess
from pathlib import Path
import wave
import contextlib
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gtts import gTTS
from supabase import create_client
from dotenv import load_dotenv
from pymongo import MongoClient
import whisper
import imageio_ffmpeg

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-backend")

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()

# FFmpeg (imageio_ffmpeg gives a reliable path if available)
try:
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"Using FFmpeg from: {FFMPEG_PATH}")
except Exception as e:
    FFMPEG_PATH = "ffmpeg"  # fallback to system ffmpeg
    logger.warning(f"Could not get ffmpeg path via imageio_ffmpeg: {e}. Falling back to '{FFMPEG_PATH}'")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "interview-audios")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in environment. Supabase calls may fail.")

# Create supabase client (will raise if keys invalid)
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception as e:
    logger.error(f"Failed to create Supabase client: {e}")
    supabase = None

MONGO_URI = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.warning("MONGO_URL / MONGO_URI not set. Mongo operations will fail.")

mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
mongo_db = mongo_client["recruiter-platform"] if mongo_client else None
candidatereg_collection = mongo_db["candidateregisters"] if mongo_db else None
interviews_collection = mongo_db["interviews"] if mongo_db else None

# ---------------------------
# App and static dir
# ---------------------------
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Interview Voice Bot Backend")

# CORS - allow all origins for now (adjust for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # change in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Whisper model (lazy load)
# ---------------------------
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model (base) ...")
        whisper_model = whisper.load_model("base", device="cpu")
        logger.info("Whisper model loaded")
    return whisper_model

# ---------------------------
# Questions and models
# ---------------------------
QUESTIONS = [
    "How many years of experience do you have?",
    "What is your current CTC?",
    "What is your expected CTC?",
    "Which is your current location?",
    "Are you open to relocation?",
    "What is your notice period?",
]

class StartRequest(BaseModel):
    email: str

class UpdateAnswerRequest(BaseModel):
    question_index: int
    new_answer: str
    status: str = "updated"

# ---------------------------
# Helpers
# ---------------------------
def convert_to_wav(input_path: str) -> str:
    """
    Ensures audio is PCM 16k mono WAV as required by Whisper.
    If the input is already correct, return input_path.
    Otherwise, use ffmpeg to convert and return new path.
    """
    try:
        with contextlib.closing(wave.open(input_path, "rb")) as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            comptype = wf.getcomptype()
            logger.debug(f"Existing WAV params: channels={channels}, framerate={framerate}, sampwidth={sampwidth}, comptype={comptype}")
            if channels == 1 and framerate == 16000 and sampwidth == 2 and comptype == "NONE":
                return input_path
    except wave.Error:
        # not a wav or not readable - proceed to convert
        logger.debug("Input not a compatible WAV, will attempt conversion")

    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-ar", "16000",       # sample rate
        "-ac", "1",           # mono
        "-c:a", "pcm_s16le",  # PCM 16-bit little endian
        output_path
    ]
    try:
        logger.info(f"Running ffmpeg: {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug(f"FFmpeg stdout: {proc.stdout[:100]}")
        return output_path
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg conversion failed: {stderr}")
        # Cleanup output if exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail="Audio conversion failed on server")

def text_to_speech(text: str, filename: str) -> str:
    """
    Save TTS to static folder and return filepath
    """
    filepath = STATIC_DIR / filename
    try:
        gTTS(text=text, lang="en").save(str(filepath))
        return str(filepath)
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed")

def upload_to_supabase(file_path: str, candidate_id: str, prefix: str = "bot_q") -> str:
    """
    Upload file to Supabase public bucket and return public URL.
    """
    if not supabase:
        logger.warning("Supabase client not configured; skipping upload.")
        return ""
    try:
        ext = os.path.splitext(file_path)[1] or ""
        path_in_bucket = f"{candidate_id}/{prefix}_{uuid.uuid4().hex}{ext}"
        logger.info(f"Uploading {file_path} to supabase at {path_in_bucket}")
        with open(file_path, "rb") as f:
            data = f.read()
            try:
                supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, data)
            except Exception as upload_err:
                # If object exists, attempt update; else bubble up
                if "exists" in str(upload_err).lower():
                    logger.info("File already exists; attempting update")
                    supabase.storage.from_(BUCKET_NAME).update(path_in_bucket, data)
                else:
                    logger.error(f"Supabase upload error: {upload_err}")
                    raise
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{path_in_bucket}"
        return public_url
    except Exception as e:
        logger.error(f"Failed to upload to supabase: {e}")
        # don't fail entire flow for analytics - return empty string so caller can decide
        return ""

def safe_get_mongo_candidate_by_email(email: str) -> Optional[dict]:
    try:
        if not candidatereg_collection:
            return None
        return candidatereg_collection.find_one({"email": email})
    except Exception as e:
        logger.error(f"Mongo error fetching candidate by email {email}: {e}")
        return None

def safe_get_mongo_candidate_by_id(candidate_id: str) -> Optional[dict]:
    try:
        if not candidatereg_collection:
            return None
        return candidatereg_collection.find_one({"_id": candidate_id})
    except Exception as e:
        logger.error(f"Mongo error fetching candidate by id {candidate_id}: {e}")
        return None

# ---------------------------
# Routes
# ---------------------------
@app.post("/start_interview")
async def start_interview(req: StartRequest):
    try:
        email = req.email.strip().lower()
        logger.info(f"start_interview called for email: {email}")

        candidate_doc = safe_get_mongo_candidate_by_email(email)
        if not candidate_doc:
            logger.warning(f"No candidate found for email: {email}")
            raise HTTPException(status_code=404, detail=f"Candidate not found for email: {email}")

        candidate_id = str(candidate_doc["_id"])
        name = candidate_doc.get("name")
        logger.info(f"Candidate found: id={candidate_id}, name={name}")

        # Supabase candidates table insert if not exists
        try:
            if supabase:
                existing = supabase.table("candidates").select("*").eq("candidate_id", candidate_id).execute()
                if not existing.data:
                    supabase.table("candidates").insert({
                        "candidate_id": candidate_id,
                        "name": name,
                        "email": email
                    }).execute()
        except Exception as e:
            logger.error(f"Supabase candidates sync failed: {e}")

        # Mongo interviews doc
        try:
            if interviews_collection and not interviews_collection.find_one({"candidate_id": candidate_id}):
                interviews_collection.insert_one({"candidate_id": candidate_id, "qa": [], "interview_finished": False})
        except Exception as e:
            logger.error(f"Mongo interviews insert failed: {e}")

        # Reset/create Supabase session
        try:
            if supabase:
                supabase.table("sessions").delete().eq("candidate_id", candidate_id).execute()
                supabase.table("sessions").insert({
                    "candidate_id": candidate_id,
                    "q_index": 0,
                    "status": "active"
                }).execute()
        except Exception as e:
            logger.error(f"Supabase sessions reset failed: {e}")

        return {
            "message": "Interview started",
            "candidate_id": candidate_id,
            "name": name,
            "email": email,
            "next_question_url": f"/question/{candidate_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("start_interview failed")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {e}")

@app.get("/question/{candidate_id}")
async def get_question(candidate_id: str):
    try:
        logger.info(f"get_question called for candidate_id: {candidate_id}")
        if not supabase:
            raise HTTPException(status_code=500, detail="Supabase client not configured")

        session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
        if not session_res.data:
            logger.warning(f"Session not found for candidate_id: {candidate_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        session = session_res.data[0]
        q_index = session.get("q_index", 0)
        logger.info(f"Session q_index: {q_index}")

        if q_index >= len(QUESTIONS):
            return {"done": True, "message": "Interview finished"}

        question = QUESTIONS[q_index]
        filename = f"{candidate_id}_q{q_index}.mp3"
        filepath = text_to_speech(question, filename)
        audio_url = upload_to_supabase(filepath, candidate_id, prefix=f"bot_q_{q_index}")

        return {
            "done": False,
            "question_index": q_index,
            "question": question,
            "audio_url": audio_url
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_question failed")
        raise HTTPException(status_code=500, detail=f"Failed to fetch question: {e}")

@app.post("/submit_answer/{candidate_id}/{question_index}")
async def submit_answer(candidate_id: str, question_index: int, file: UploadFile = File(...)):
    tmp_input = None
    tmp_wav_path = None
    try:
        logger.info(f"submit_answer called candidate_id={candidate_id} question_index={question_index} filename={file.filename}")
        if not supabase:
            logger.warning("Supabase client not configured; continuing but uploads will be skipped")

        # validate session exists
        try:
            session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute() if supabase else None
            if supabase and (not session_res or not session_res.data):
                logger.warning("Session not found in supabase")
                raise HTTPException(status_code=404, detail="Session not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking session: {e}")
            raise HTTPException(status_code=500, detail="Failed to validate session")

        # Save uploaded file to temp
        suffix = os.path.splitext(file.filename)[1] or ".webm"
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        content = await file.read()
        tmp_input.write(content)
        tmp_input.flush()
        tmp_input.close()
        logger.info(f"Saved uploaded audio to {tmp_input.name}")

        # Convert to wav suitable for whisper
        tmp_wav_path = convert_to_wav(tmp_input.name)

        text_answer = "(Transcription failed)"
        status = "error"

        # Transcribe with Whisper
        try:
            model = get_whisper_model()
            audio = whisper.load_audio(tmp_wav_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)
            text_answer = result.text.strip() or "(Could not detect speech)"
            status = "ok"
            logger.info(f"Transcription result: {text_answer[:200]}")
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")

        # Upload original audio (not converted one) so we preserve original format
        audio_url = ""
        try:
            audio_url = upload_to_supabase(tmp_input.name, candidate_id, prefix=f"answer_{question_index}")
        except Exception as e:
            logger.error(f"Upload to supabase failed: {e}")

        # Insert into supabase interviews table (if configured)
        try:
            if supabase:
                supabase.table("interviews").insert({
                    "candidate_id": candidate_id,
                    "question": QUESTIONS[question_index] if 0 <= question_index < len(QUESTIONS) else f"Q{question_index}",
                    "answer_text": text_answer,
                    "status": status,
                    "answer_audio_url": audio_url
                }).execute()
        except Exception as e:
            logger.error(f"Supabase interviews insert failed: {e}")

        # Save to Mongo
        try:
            if interviews_collection:
                interviews_collection.update_one(
                    {"candidate_id": candidate_id},
                    {"$push": {"qa": [{"question": QUESTIONS[question_index] if 0 <= question_index < len(QUESTIONS) else f"Q{question_index}", "answer": text_answer, "audio_url": audio_url}]}}
                )
        except Exception as e:
            logger.error(f"Mongo update failed: {e}")

        # Advance session index
        try:
            if supabase and session_res and session_res.data:
                new_index = session_res.data[0].get("q_index", 0) + 1
                supabase.table("sessions").update({"q_index": new_index}).eq("candidate_id", candidate_id).execute()
        except Exception as e:
            logger.error(f"Failed to advance session q_index: {e}")

        return {"answer_text": text_answer, "status": status, "next_question_url": f"/question/{candidate_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("submit_answer failed")
        raise HTTPException(status_code=500, detail=f"Failed to submit answer: {e}")
    finally:
        # cleanup temp files
        try:
            if tmp_input and os.path.exists(tmp_input.name):
                os.remove(tmp_input.name)
                logger.debug(f"Removed temp input {tmp_input.name}")
            if tmp_wav_path and os.path.exists(tmp_wav_path) and tmp_wav_path != (tmp_input.name if tmp_input else None):
                os.remove(tmp_wav_path)
                logger.debug(f"Removed converted wav {tmp_wav_path}")
        except Exception as cleanup_err:
            logger.warning(f"Temp cleanup error: {cleanup_err}")

@app.post("/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: str):
    try:
        logger.info(f"finish_interview called for {candidate_id}")
        candidate_doc = safe_get_mongo_candidate_by_id(candidate_id)
        name = candidate_doc.get("name") if candidate_doc else None
        email = candidate_doc.get("email") if candidate_doc else None

        try:
            if interviews_collection:
                interviews_collection.update_one(
                    {"candidate_id": candidate_id},
                    {"$set": {"interview_finished": True}}
                )
        except Exception as e:
            logger.error(f"Mongo update error on finish_interview: {e}")

        try:
            if supabase:
                supabase.table("sessions").update({"status": "finished"}).eq("candidate_id", candidate_id).execute()
        except Exception as e:
            logger.error(f"Supabase update sessions failed: {e}")

        return {
            "message": "Interview finished",
            "candidate_id": candidate_id,
            "name": name,
            "email": email,
            "summary_url": f"/get_answers/{candidate_id}"
        }
    except Exception as e:
        logger.exception("finish_interview failed")
        raise HTTPException(status_code=500, detail=f"Failed to finish interview: {e}")

@app.get("/get_answers/{candidate_id}")
async def get_answers(candidate_id: str):
    try:
        logger.info(f"get_answers called for {candidate_id}")
        mongo_doc = None
        supa_res = None
        candidate_doc = None

        try:
            if interviews_collection:
                mongo_doc = interviews_collection.find_one({"candidate_id": candidate_id}, {"_id": 0})
        except Exception as e:
            logger.error(f"Mongo get_answers error: {e}")

        try:
            if supabase:
                supa_res = supabase.table("interviews").select("*").eq("candidate_id", candidate_id).execute()
        except Exception as e:
            logger.error(f"Supabase get_answers error: {e}")

        try:
            candidate_doc = safe_get_mongo_candidate_by_id(candidate_id)
        except Exception as e:
            logger.error(f"Candidate lookup error: {e}")

        return {
            "candidate_id": candidate_id,
            "name": candidate_doc.get("name") if candidate_doc else None,
            "email": candidate_doc.get("email") if candidate_doc else None,
            "qa_mongo": mongo_doc.get("qa", []) if mongo_doc else [],
            "qa_supabase": supa_res.data if supa_res and hasattr(supa_res, "data") else []
        }
    except Exception as e:
        logger.exception("get_answers failed")
        raise HTTPException(status_code=500, detail=f"Failed to fetch answers: {e}")

@app.put("/update_answer/{candidate_id}")
async def update_answer(candidate_id: str, req: UpdateAnswerRequest = Body(...)):
    try:
        logger.info(f"update_answer called for {candidate_id} question_index={req.question_index}")
        question = QUESTIONS[req.question_index] if 0 <= req.question_index < len(QUESTIONS) else None
        if question is None:
            raise HTTPException(status_code=400, detail="Invalid question_index")

        # Mongo update
        try:
            if interviews_collection:
                interviews_collection.update_one(
                    {"candidate_id": candidate_id, "qa.question": question},
                    {"$set": {
                        "qa.$.answer": req.new_answer,
                        "qa.$.status": req.status
                    }}
                )
        except Exception as e:
            logger.error(f"Mongo update failed in update_answer: {e}")

        # Supabase update
        try:
            if supabase:
                supabase.table("interviews").update({
                    "answer_text": req.new_answer,
                    "status": req.status
                }).eq("candidate_id", candidate_id).eq("question", question).execute()
        except Exception as e:
            logger.error(f"Supabase update failed in update_answer: {e}")

        candidate_doc = safe_get_mongo_candidate_by_id(candidate_id)
        name = candidate_doc.get("name") if candidate_doc else None
        email = candidate_doc.get("email") if candidate_doc else None

        return {
            "message": "Answer updated successfully",
            "candidate_id": candidate_id,
            "name": name,
            "email": email,
            "question": question,
            "updated_answer": req.new_answer,
            "status": req.status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("update_answer failed")
        raise HTTPException(status_code=500, detail=f"Failed to update answer: {e}")

# ---------------------------
# Static files
# ---------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root for health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "Interview Voice Bot Backend running"}
