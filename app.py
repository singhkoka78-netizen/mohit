import os
import uuid
import tempfile
import subprocess
from pathlib import Path
import wave
import contextlib
import traceback
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

load_dotenv()
# FFmpeg Path
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using FFmpeg from: {FFMPEG_PATH}")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "interview-audios")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MONGO_URI = os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["recruiter-platform"]
candidatereg_collection = mongo_db["candidateregisters"]
interviews_collection = mongo_db["interviews"]

# App
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app = FastAPI(title="Interview Voice Bot Backend")

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# whisper
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base", device="cpu")
        print("Whisper model ready")
    return whisper_model

QUESTIONS = [
    "How many years of experience do you have?",
    "What is your current CTC?",
    "What is your expected CTC?",
    "Which is your current location?",
    "Are you open to relocation?",
    "What is your notice period?",
]

#  model
class StartRequest(BaseModel):
    email: str

def convert_to_wav(input_path: str) -> str:
    # Convert audio to WAV 16kHz mono PCM16
    try:
        with contextlib.closing(wave.open(input_path, "rb")) as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            comptype = wf.getcomptype()
            if channels == 1 and framerate == 16000 and sampwidth == 2 and comptype == "NONE":
                return input_path
    except wave.Error:
        pass  

    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [FFMPEG_PATH, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return output_path

def text_to_speech(text: str, filename: str) -> str:
    filepath = STATIC_DIR / filename
    gTTS(text=text, lang="en").save(str(filepath))
    return str(filepath)

def upload_to_supabase(file_path: str, candidate_id: str, prefix="bot_q") -> str:
    path_in_bucket = f"{candidate_id}/{prefix}_{uuid.uuid4().hex}{os.path.splitext(file_path)[1]}"
    with open(file_path, "rb") as f:
        try:
            supabase.storage.from_(BUCKET_NAME).upload(path_in_bucket, f.read())
        except Exception as e:
            if "exists" in str(e).lower():
                with open(file_path, "rb") as f2:
                    supabase.storage.from_(BUCKET_NAME).update(path_in_bucket, f2.read())
            else:
                raise
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{path_in_bucket}"

# --------------------------
# Start Interview (with Welcome)
# --------------------------
@app.post("/start_interview")
async def start_interview(req: StartRequest):
    try:
        print("👉 /start_interview called with:", req.dict())

        email = req.email.strip().lower()
        print(f"🔍 Looking up candidate: {email}")
        candidate_doc = candidatereg_collection.find_one({"email": email})
        if not candidate_doc:
            print("❌ Candidate not found in Mongo")
            raise HTTPException(404, f"Candidate not found for email: {email}")

        candidate_id = str(candidate_doc["_id"])
        name = candidate_doc.get("name", "Candidate")
        print(f"✅ Candidate found: {candidate_id}, {name}")

        # Supabase candidates
        try:
            existing = supabase.table("candidates").select("*").eq("candidate_id", candidate_id).execute()
            if not existing.data:
                supabase.table("candidates").insert({
                    "candidate_id": candidate_id,
                    "name": name,
                    "email": email
                }).execute()
        except Exception as e:
            print("❌ Supabase error:", str(e))
            traceback.print_exc()
            raise

        # Mongo interviews
        if not interviews_collection.find_one({"candidate_id": candidate_id}):
            interviews_collection.insert_one({"candidate_id": candidate_id, "qa": [], "interview_finished": False})

        # Reset Supabase session
        try:
            supabase.table("sessions").delete().eq("candidate_id", candidate_id).execute()
            supabase.table("sessions").insert({
                "candidate_id": candidate_id,
                "q_index": 0,
                "status": "active"
            }).execute()
        except Exception as e:
            print("❌ Supabase session error:", str(e))
            traceback.print_exc()
            raise

        # 👇 Generate Welcome Message TTS
        welcome_text = f"Welcome {name}, let's begin your interview."
        welcome_filename = f"{candidate_id}_welcome.mp3"
        welcome_filepath = text_to_speech(welcome_text, welcome_filename)
        welcome_audio_url = upload_to_supabase(welcome_filepath, candidate_id, prefix="welcome")

        return {
            "message": "Interview started",
            "candidate_id": candidate_id,
            "candidate_name": name,
            "welcome_audio_url": welcome_audio_url,
            "next_question_url": f"/question/{candidate_id}"
        }

    except Exception as e:
        print("💥 ERROR in /start_interview:", str(e))
        traceback.print_exc()
        raise HTTPException(500, f"Failed to start interview: {e}")

# --------------------------
# Get Question
# --------------------------
@app.get("/question/{candidate_id}")
async def get_question(candidate_id: str):
    try:
        session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
        if not session_res.data:
            raise HTTPException(404, "Session not found")
        session = session_res.data[0]
        q_index = session["q_index"]
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
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch question: {e}")

# --------------------------
# Submit Answer
# --------------------------
@app.post("/submit_answer/{candidate_id}/{currentQuestionIndex}")
async def submit_answer(candidate_id: str, currentQuestionIndex: int, file: UploadFile = File(...)):
    tmp_input = tmp_wav_path = None
    try:
        # check session
        session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
        if not session_res.data:
            raise HTTPException(404, "Session not found")
        session = session_res.data[0]

        # temp audio
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        tmp_input.write(await file.read())
        tmp_input.close()
        tmp_wav_path = convert_to_wav(tmp_input.name)

        # transcription
        text_answer = "(Transcription failed)"
        status = "error"
        try:
            model = get_whisper_model()
            audio = whisper.load_audio(tmp_wav_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)
            text_answer = result.text.strip() or "(Could not detect speech)"
            status = "ok"
        except Exception as e:
            print("Whisper error:", e)

        # Upload audio to Supabase
        audio_url = upload_to_supabase(tmp_input.name, candidate_id, prefix=f"answer_{currentQuestionIndex}")

        supabase.table("interviews").insert({
            "candidate_id": candidate_id,
            "question": QUESTIONS[currentQuestionIndex],
            "answer_text": text_answer,
            "status": status,
            "answer_audio_url": audio_url
        }).execute()

        # save in Mongo 
        interviews_collection.update_one(
            {"candidate_id": candidate_id},
            {"$push": {"qa": [{"question": QUESTIONS[currentQuestionIndex], "answer": text_answer, "audio_url": audio_url}]} }
        )

        # update session index
        supabase.table("sessions").update({"q_index": session["q_index"] + 1}).eq("candidate_id", candidate_id).execute()

        return {"answer_text": text_answer, "status": status, "next_question_url": f"/question/{candidate_id}"}

    finally:
        if tmp_input and os.path.exists(tmp_input.name):
            os.remove(tmp_input.name)
        if tmp_wav_path and os.path.exists(tmp_wav_path) and tmp_wav_path != tmp_input.name:
            os.remove(tmp_wav_path)

# --------------------------
# Finish Interview
# --------------------------
@app.post("/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: str):
    try:
        # update Mongo
        interviews_collection.update_one(
            {"candidate_id": candidate_id},
            {"$set": {"interview_finished": True}}
        )
        # supabase session updation
        supabase.table("sessions").update({"status": "finished"}).eq("candidate_id", candidate_id).execute()
        return {"message": "Interview finished", "candidate_id": candidate_id, "summary_url": f"/get_answers/{candidate_id}"}
    except Exception as e:
        raise HTTPException(500, f"Failed to finish interview: {e}")

# --------------------------
# Get Answers
# --------------------------
@app.get("/get_answers/{candidate_id}")
async def get_answers(candidate_id: str):
    try:
        mongo_doc = interviews_collection.find_one({"candidate_id": candidate_id}, {"_id": 0})
        supa_res = supabase.table("interviews").select("*").eq("candidate_id", candidate_id).execute()
        return {
            "candidate_id": candidate_id,
            "qa_mongo": mongo_doc.get("qa", []) if mongo_doc else [],
            "qa_supabase": supa_res.data
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch answers: {e}")

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")
