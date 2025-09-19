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

# FFmpeg via imageio-ffmpeg
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
print("🎬 Using FFmpeg from imageio:", FFMPEG_PATH)

# Setup Supabase
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "interview-audios")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Setup MongoDB
MONGO_URI = os.getenv("MONGO_URL")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["recruiter-platform"]
candidates_collection = mongo_db["candidates"]
interviews_collection = mongo_db["interviews"]

# App Config
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# ✅ Add root_path="/api" so all routes are under /api/
app = FastAPI(title="Interview Voice Bot Backend", root_path="/api")

# ✅ Configure CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://mohit-7t2w.onrender.com"],  # frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper (lazy load)
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("⏳ Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base", device="cpu")  # force CPU
        print("✅ Whisper model ready")
    return whisper_model

# Questions
QUESTIONS = [
    "How many years of experience do you have?",
    "What is your current CTC?",
    "What is your expected CTC?",
    "Which is your current location?",
    "Are you open to relocation?",
    "What is your notice period?",
]

class StartRequest(BaseModel):
    candidate_id: str
    name: str | None = None
    email: str | None = None

# Helper: Convert WebM → WAV
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

# Routes
@app.post("/start_interview")
async def start_interview(req: StartRequest):
    # Check if candidate already exists
    existing = supabase.table("candidates").select("*").eq("email", req.email).execute()
    if existing.data:
        candidate_id = existing.data[0]["id"]   # 👈 use database auto id
    else:
        inserted = supabase.table("candidates").insert({
            "name": req.name,
            "email": req.email
        }).execute()
        candidate_id = inserted.data[0]["id"]   # 👈 fetch auto-generated id
 
        # Save in Mongo too
        candidates_collection.insert_one({
            "_id": candidate_id,
            "candidate_id": candidate_id,
            "name": req.name,
            "email": req.email
        })
        print(f"✅ Candidate saved in Mongo: {candidate_id}, {req.name}, {req.email}")
 
    # Start session
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

@app.get("/question/{candidate_id}")
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

    # Generate TTS if not exists
    if not filepath.exists():
        gTTS(text=question, lang="en").save(str(filepath))

    # ✅ Upload to Supabase bucket if not already uploaded
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

@app.post("/submit_answer/{candidate_id}/{question_index}")
async def submit_answer(candidate_id: str, question_index: int, file: UploadFile = File(...)):
    candidate_id = candidate_id.strip()  # ✅ ensure clean ID

    # 1. Verify candidate exists in Supabase
    candidate_check = supabase.table("candidates").select("candidate_id").eq("candidate_id", candidate_id).execute()
    if not candidate_check.data:
        raise HTTPException(404, f"Candidate {candidate_id} not found in Supabase")

    # 2. Get session
    session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
    if not session_res.data:
        raise HTTPException(404, "Session not found")

    session = session_res.data[0]

    # 3. Save uploaded audio temporarily
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp_input.write(await file.read())
    tmp_input.close()

    # 4. Convert WebM → WAV
    tmp_wav_path = convert_to_wav(tmp_input.name)

    # 5. Transcribe with Whisper
    text_answer = ""
    status = "error"
    try:
        model = get_whisper_model()
        audio = whisper.load_audio(tmp_wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        text_answer = result.text.strip() or "(Could not detect speech)"
        status = "ok" if result.text.strip() else "error"
    except Exception as e:
        print("❌ Whisper error:", e)
        text_answer = "(Transcription failed)"
        status = "error"

    # 6. Upload original audio to Supabase Storage
    path_in_bucket = f"{candidate_id}/{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
    with open(tmp_input.name, "rb") as f:
        supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read())
    audio_url = supabase.storage.from_(BUCKET_NAME).get_public_url(path_in_bucket)

    # 7. Cleanup temp files
    os.remove(tmp_input.name)
    os.remove(tmp_wav_path)

    # 8. Save answer in Supabase
    supabase.table("interviews").insert({
        "candidate_id": candidate_id,
        "question": QUESTIONS[question_index],
        "answer_text": text_answer,
        "status": status,
        "answer_audio_url": audio_url
    }).execute()

    # 9. Save in Mongo too
    interviews_collection.insert_one({
        "candidate_id": candidate_id,
        "question_index": question_index,
        "question": QUESTIONS[question_index],
        "answer_text": text_answer,
        "status": status,
        "answer_audio_url": audio_url
    })

    print(f"✅ Answer saved for candidate {candidate_id}, Q{question_index}")

    # 10. Advance session if transcript is valid
    if status == "ok":
        supabase.table("sessions").update(
            {"q_index": session["q_index"] + 1}
        ).eq("candidate_id", candidate_id).execute()

    return {
        "answer_text": text_answer,
        "status": status,
        "next_question_url": f"/api/question/{candidate_id}"
    }

@app.get("/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: str):
    res = supabase.table("interviews").select("*").eq("candidate_id", candidate_id.strip()).execute()
    supabase.table("sessions").delete().eq("candidate_id", candidate_id.strip()).execute()
    return {"candidate_id": candidate_id.strip(), "answers": res.data}

# NEW ROUTE: Fetch all questions + answers as transcript
@app.get("/get_answers/{candidate_id}")
async def get_answers(candidate_id: str):
    clean_id = candidate_id.strip()  # remove spaces/newlines
    res = supabase.table("interviews").select("question, answer_text").eq("candidate_id", clean_id).execute()

    if not res.data:
        return {"candidate_id": clean_id, "answers": []}

    # Make Q&A transcript
    transcript = []
    for row in res.data:
        transcript.append(f"Q: {row['question']}")
        transcript.append(f"A: {row['answer_text']}")

    return {
        "candidate_id": clean_id,
        "answers": res.data,     # raw rows
        "transcript": transcript # formatted Q&A
    }

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
