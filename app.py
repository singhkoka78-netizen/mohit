import os
import uuid
import tempfile
import subprocess
from pathlib import Path
import wave

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
        whisper_model = whisper.load_model("tiny", device="cpu")  # lighter for Render
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
    name: str | None = None
    email: str | None = None
    candidate_id: str | None = None   # ✅ NEW (optional)

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
    # ✅ If candidate_id provided, use it
    if req.candidate_id:
        candidate_id = req.candidate_id
    else:
        # otherwise, check by email or generate new
        existing = supabase.table("candidates").select("*").eq("email", req.email).execute()
        if existing.data:
            candidate_id = existing.data[0]["candidate_id"]
        else:
            candidate_id = str(uuid.uuid4())
            supabase.table("candidates").insert({
                "candidate_id": candidate_id,
                "name": req.name,
                "email": req.email
            }).execute()

            candidates_collection.insert_one({
                "_id": candidate_id,
                "candidate_id": candidate_id,
                "name": req.name,
                "email": req.email
            })
            print(f"✅ Candidate saved in Mongo: {candidate_id}, {req.name}, {req.email}")

    # Create session
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
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(404, "Session not found")

    session = session_res.data[0]
    idx = session["q_index"]

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
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(404, "Session not found")
    session = session_res.data[0]

    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp_input.write(await file.read())
    tmp_input.close()

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

    path_in_bucket = f"{candidate_id}/{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
    with open(tmp_input.name, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read())
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    os.remove(tmp_input.name)
    os.remove(tmp_wav_path)

    qa_pair = [
        f"Q: {QUESTIONS[question_index]}",
        f"A: {text_answer}"
    ]
    interviews_collection.update_one(
        {"candidate_id": candidate_id},
        {"$push": {"qa": qa_pair}},
        upsert=True
    )
    print(f"✅ Answer saved in Mongo for candidate {candidate_id}, Q{question_index}")

    if status == "ok":
        supabase.table("sessions").update(
            {"q_index": session["q_index"] + 1}
        ).eq("candidate_id", candidate_id).execute()

    return {
        "answer_text": text_answer,
        "status": status,
        "saved_in_mongo": True,
        "next_question_url": f"/api/question/{candidate_id}"
    }

@app.get("/api/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: str):
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
    for q, a in qa_list:
        transcript.append(q)
        transcript.append(a)

    return {
        "candidate_id": clean_id,
        "answers": qa_list,
        "transcript": transcript
    }

# ------------------------------
# Static files
# ------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
