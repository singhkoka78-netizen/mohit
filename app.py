import os
import uuid
import tempfile
import subprocess
from pathlib import Path
import wave
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gtts import gTTS
from supabase import create_client
from dotenv import load_dotenv
from pymongo import MongoClient
import whisper
import imageio_ffmpeg

# ------------------------------
# FFmpeg via imageio-ffmpeg
# ------------------------------
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
print("🎬 Using FFmpeg from imageio:", FFMPEG_PATH)

def load_audio_with_ffmpeg(path: str, sr: int = 16000):
    import numpy as np
    cmd = [
        FFMPEG_PATH,
        "-nostdin",
        "-threads", "0",
        "-i", path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

# ------------------------------
# Setup Supabase
# ------------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "interview-audios")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------
# Setup MongoDB
# ------------------------------
MONGO_URI = os.getenv("MONGO_URL")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["recruiter-platform"]
candidates_collection = mongo_db["candidates"]
interviews_collection = mongo_db["interviews"]

# ------------------------------
# App Config
# ------------------------------
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app = FastAPI(title="Interview Voice Bot Backend")

# ------------------------------
# CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Whisper (lazy load)
# ------------------------------
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("⏳ Loading Whisper model (tiny)...")
        whisper_model = whisper.load_model("tiny", device="cpu")  # change device if you have GPU
        print("✅ Whisper model ready")
    return whisper_model

# ------------------------------
# Questions
# ------------------------------
QUESTIONS = [
    "How many years of experience do you have?",
    "What is your current CTC?",
    "What is your expected CTC?",
    "Which is your current location?",
    "Are you open to relocation?",
    "What is your notice period?",
]

class StartRequest(BaseModel):
    # You can pass candidate_id (re-use), or name/email to create/find by email
    candidate_id: str | None = None
    name: str | None = None
    email: str | None = None

# ------------------------------
# Helper: Convert WebM → WAV
# ------------------------------
def convert_to_wav(input_path: str) -> str:
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    try:
        subprocess.run(
            [FFMPEG_PATH, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("✅ FFmpeg conversion OK:", output_path)
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg failed:", e.stderr.decode())
        raise
    return output_path

# ------------------------------
# Routes (under /api)
# ------------------------------
@app.get("/")
def root():
    return {"message": "Backend running 🚀"}

@app.post("/api/start_interview")
async def start_interview(req: StartRequest):
    """
    If `candidate_id` is provided in request body, we validate and reuse it.
    Otherwise, fallback to lookup by email and create candidate if not present.
    Returns candidate_id and next_question url.
    """
    candidate_id = None

    # 1) If client provided candidate_id -> validate exists in Mongo (preferred flow)
    if req.candidate_id:
        candidate_id = req.candidate_id.strip()
        candidate = candidates_collection.find_one({"candidate_id": candidate_id})
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate id provided but not found")
    else:
        # 2) Fallback: lookup by email in Supabase; if not found create new candidate
        if not req.email:
            raise HTTPException(status_code=400, detail="Either candidate_id or email is required")

        existing = supabase.table("candidates").select("*").eq("email", req.email).execute()
        if existing.data:
            candidate_id = existing.data[0]["candidate_id"]
        else:
            candidate_id = str(uuid.uuid4())
            # Insert into Supabase
            supabase.table("candidates").insert({
                "candidate_id": candidate_id,
                "name": req.name,
                "email": req.email
            }).execute()

            # Insert into Mongo
            candidates_collection.insert_one({
                "_id": candidate_id,
                "candidate_id": candidate_id,
                "name": req.name,
                "email": req.email
            })
            print(f"✅ Candidate saved in Mongo: {candidate_id}, {req.name}, {req.email}")

    # Create a session row for interview start (if one exists you may want to reset or upsert)
    supabase.table("sessions").insert({
        "candidate_id": candidate_id,
        "q_index": 0
    }).execute()

    return {
        "message": "Interview started",
        "candidate_id": candidate_id,
        "total_questions": len(QUESTIONS),
        "next_question_url": f"/api/question/{candidate_id}"
    }

@app.get("/api/question/{candidate_id}")
async def get_question(candidate_id: str):
    # Ensure session exists
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = session_res.data[0]
    idx = session.get("q_index", 0)

    if idx >= len(QUESTIONS):
        return {"done": True, "message": "Interview finished"}

    question = QUESTIONS[idx]
    filename = f"q_{candidate_id}_{idx}.mp3"
    filepath = STATIC_DIR / filename

    if not filepath.exists():
        gTTS(text=question, lang="en").save(str(filepath))

    path_in_bucket = f"{candidate_id}/bot_q_{idx}.mp3"
    with open(filepath, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read(), {"upsert": "true"})
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    return {
        "done": False,
        "question_index": idx,
        "question": question,
        "audio_url": audio_url
    }

@app.post("/api/submit_answer/{candidate_id}/{question_index}")
async def submit_answer(candidate_id: str, question_index: int, file: UploadFile = File(...)):
    # Validate session exists
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(status_code=404, detail="Session not found")
    session = session_res.data[0]

    # Save uploaded file temporarily
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp_input.write(await file.read())
    tmp_input.close()

    # Convert to wav for whisper
    tmp_wav_path = convert_to_wav(tmp_input.name)

    text_answer = ""
    status = "error"
    try:
        model = get_whisper_model()
        audio = load_audio_with_ffmpeg(tmp_wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        text_answer = result.text.strip()

        if text_answer:
            status = "ok"
        else:
            text_answer = "(Could not detect speech)"
    except Exception as e:
        print("❌ Whisper error:", e)
        text_answer = "(Transcription failed)"

    # Upload original recorded file to Supabase storage
    path_in_bucket = f"{candidate_id}/{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
    with open(tmp_input.name, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read())
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    # Cleanup tmp files
    try:
        os.remove(tmp_input.name)
    except Exception:
        pass
    try:
        os.remove(tmp_wav_path)
    except Exception:
        pass

    # Save as structured object in Mongo
    qa_obj = {
        "question_index": question_index,
        "question": QUESTIONS[question_index],
        "answer": text_answer,
        "audio_url": audio_url,
        "ts": datetime.utcnow().isoformat()
    }

    interviews_collection.update_one(
        {"candidate_id": candidate_id},
        {"$push": {"qa": qa_obj}},
        upsert=True
    )
    print(f"✅ Answer saved in Mongo for candidate {candidate_id}, Q{question_index}")

    # advance session index only if transcription ok (you can change logic if you want)
    if status == "ok":
        supabase.table("sessions").update(
            {"q_index": session.get("q_index", 0) + 1}
        ).eq("candidate_id", candidate_id).execute()

    return {
        "answer_text": text_answer,
        "status": status,
        "saved_in_mongo": True,
        "audio_url": audio_url,
        "next_question_url": f"/api/question/{candidate_id}"
    }

@app.get("/api/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: str):
    # remove session
    supabase.table("sessions").delete().eq("candidate_id", candidate_id.strip()).execute()
    doc = interviews_collection.find_one({"candidate_id": candidate_id.strip()}, {"_id": 0})
    return {"candidate_id": candidate_id.strip(), "answers": doc.get("qa", []) if doc else []}

@app.get("/api/get_answers/{candidate_id}")
async def get_answers(candidate_id: str):
    clean_id = candidate_id.strip()
    doc = interviews_collection.find_one({"candidate_id": clean_id}, {"_id": 0})
    if not doc:
        return {"candidate_id": clean_id, "answers": [], "transcript": []}

    qa_list = doc.get("qa", [])
    transcript = []
    # each qa item is now an object/dict {question, answer, ...}
    for qa in qa_list:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        transcript.append(f"Q: {q}")
        transcript.append(f"A: {a}")

    return {
        "candidate_id": clean_id,
        "answers": qa_list,
        "transcript": transcript
    }

# ------------------------------
# Static files
# ------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
