# ------------------------------------------
# Complete Whisper API Server with ngrok
# ------------------------------------------

# 1️⃣ Install dependencies
!pip install -q faster-whisper fastapi uvicorn pyngrok python-multipart

# 2️⃣ Import modules
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile
import shutil, uuid, os, threading
import uvicorn
from pyngrok import ngrok

# 3️⃣ Set your ngrok authtoken (replace with yours)
NGROK_AUTHTOKEN = "3BO6NgzBWTn0gBXCHO81bDk5hVx_3qjUA87qNjuBv1qoaSVb9"
ngrok.set_auth_token(NGROK_AUTHTOKEN)

# 4️⃣ Load Whisper model (large-v3 on T4 GPU)
MODEL_SIZE = "large-v3"
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"✅ Model '{MODEL_SIZE}' loaded successfully on GPU")

# 5️⃣ Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"status": "Whisper API running"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_path = f"/content/{file_id}.wav"

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Transcribe
    segments, info = model.transcribe(file_path)
    text = " ".join([segment.text for segment in segments])

    # Cleanup
    os.remove(file_path)

    return {"text": text.strip(), "language": info.language, "duration": info.duration}

# 6️⃣ Start FastAPI server in a separate thread
def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run)
thread.start()

# 7️⃣ Create public ngrok URL
public_url = ngrok.connect(8000)
print("🌍 Public API URL:", public_url)
print("ℹ️ You can now POST audio files to /transcribe")