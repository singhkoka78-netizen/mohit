import os
import uuid
import tempfile
import subprocess
from pathlib import Path
import wave
import contextlib

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

class UpdateAnswerRequest(BaseModel):
    question_index: int
    new_answer: str
    status: str = "updated"

def convert_to_wav(input_path: str) -> str:
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


@app.post("/start_interview")
async def start_interview(req: StartRequest):
    try:
        email = req.email.strip().lower()
        candidate_doc = candidatereg_collection.find_one({"email": email})
        if not candidate_doc:
            raise HTTPException(404, f"Candidate not found for email: {email}")

        candidate_id = str(candidate_doc["_id"])
        name = candidate_doc.get("name")
    
        # Supabase candidates
        existing = supabase.table("candidates").select("*").eq("candidate_id", candidate_id).execute()
        if not existing.data:
            supabase.table("candidates").insert({
                "candidate_id": candidate_id,
                "name": name,
                "email": email
            }).execute()

        # Mongo interviews
        if not interviews_collection.find_one({"candidate_id": candidate_id}):
            interviews_collection.insert_one({"candidate_id": candidate_id, "qa": [], "interview_finished": False})

        # Reset Supabase session
        supabase.table("sessions").delete().eq("candidate_id", candidate_id).execute()
        supabase.table("sessions").insert({
            "candidate_id": candidate_id,
            "q_index": 0,
            "status": "active"
        }).execute()

        return {
            "message": "Interview started",
            "candidate_id": candidate_id,
            "name": name,
            "email": email,
            "next_question_url": f"/question/{candidate_id}"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to start interview: {e}")

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

@app.post("/submit_answer/{candidate_id}/{question_index}")
async def submit_answer(candidate_id: str, question_index: int, file: UploadFile = File(...)):
    tmp_input = tmp_wav_path = None
    try:
        session_res = supabase.table("sessions").select("*").eq("candidate_id", candidate_id).execute()
        if not session_res.data:
            raise HTTPException(404, "Session not found")
        session = session_res.data[0]

        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        tmp_input.write(await file.read())
        tmp_input.close()
        tmp_wav_path = convert_to_wav(tmp_input.name)

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

        audio_url = upload_to_supabase(tmp_input.name, candidate_id, prefix=f"answer_{question_index}")

        supabase.table("interviews").insert({
            "candidate_id": candidate_id,
            "question": QUESTIONS[question_index],
            "answer_text": text_answer,
            "status": status,
            "answer_audio_url": audio_url
        }).execute()

        interviews_collection.update_one(
            {"candidate_id": candidate_id},
            {"$push": {"qa": [{"question": QUESTIONS[question_index], "answer": text_answer, "audio_url": audio_url}]}}
        )

        supabase.table("sessions").update({"q_index": session["q_index"] + 1}).eq("candidate_id", candidate_id).execute()

        return {"answer_text": text_answer, "status": status, "next_question_url": f"/question/{candidate_id}"}

    finally:
        if tmp_input and os.path.exists(tmp_input.name):
            os.remove(tmp_input.name)
        if tmp_wav_path and os.path.exists(tmp_wav_path) and tmp_wav_path != tmp_input.name:
            os.remove(tmp_wav_path)

@app.post("/finish_interview/{candidate_id}")
async def finish_interview(candidate_id: str):
    try:
        candidate_doc = candidatereg_collection.find_one({"_id": candidate_id})
        name = candidate_doc.get("name") if candidate_doc else None
        email = candidate_doc.get("email") if candidate_doc else None

        interviews_collection.update_one(
            {"candidate_id": candidate_id},
            {"$set": {"interview_finished": True}}
        )
        supabase.table("sessions").update({"status": "finished"}).eq("candidate_id", candidate_id).execute()
        return {
            "message": "Interview finished",
            "candidate_id": candidate_id,
            "name": name,
            "email": email,
            "summary_url": f"/get_answers/{candidate_id}"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to finish interview: {e}")

@app.get("/get_answers/{candidate_id}")
async def get_answers(candidate_id: str):
    try:
        mongo_doc = interviews_collection.find_one({"candidate_id": candidate_id}, {"_id": 0})
        supa_res = supabase.table("interviews").select("*").eq("candidate_id", candidate_id).execute()
        candidate_doc = candidatereg_collection.find_one({"_id": candidate_id})

        return {
            "candidate_id": candidate_id,
            "name": candidate_doc.get("name") if candidate_doc else None,
            "email": candidate_doc.get("email") if candidate_doc else None,
            "qa_mongo": mongo_doc.get("qa", []) if mongo_doc else [],
            "qa_supabase": supa_res.data
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch answers: {e}")

# 🔥 NEW CONTROLLER: Update an answer
@app.put("/update_answer/{candidate_id}")
async def update_answer(candidate_id: str, req: UpdateAnswerRequest = Body(...)):
    try:
        question = QUESTIONS[req.question_index]

        # Mongo update
        interviews_collection.update_one(
            {"candidate_id": candidate_id, "qa.question": question},
            {"$set": {
                "qa.$.answer": req.new_answer,
                "qa.$.status": req.status
            }}
        )

        # Supabase update
        supabase.table("interviews").update({
            "answer_text": req.new_answer,
            "status": req.status
        }).eq("candidate_id", candidate_id).eq("question", question).execute()

        candidate_doc = candidatereg_collection.find_one({"_id": candidate_id})
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

    except Exception as e:
        raise HTTPException(500, f"Failed to update answer: {e}")

app.mount("/static", StaticFiles(directory="static"), name="static")
