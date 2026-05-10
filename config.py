"""Configuration for the QnA Avatar Bot.

Single-server PDF question-answering kiosk with RAG (FAISS + chunking),
Whisper STT, Gemini chat + TTS, and Edge TTS fallback.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# SERVER
# ============================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("QNA_PORT", "5005"))


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDpNctRMQO7Uxlvjyce3FwTSFbttIkFMkU")

# Chat / RAG model
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

# Embeddings model for FAISS
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

# Native Gemini TTS (preview).  Falls back to Edge TTS on failure.
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
GEMINI_TTS_VOICE_EN = os.environ.get("GEMINI_TTS_VOICE_EN", "Kore")   # female
GEMINI_TTS_VOICE_UR = os.environ.get("GEMINI_TTS_VOICE_UR", "Kore")


# ============================================================
# AVATAR
# ============================================================

AVATAR_URL = (
    "https://raw.githubusercontent.com/met4citizen/TalkingHead/refs/heads/main/avatars/brunette.glb"
)
AVATAR_FALLBACK = (
    "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@1.7/avatars/brunette.glb"
)
AVATAR_NAME = "QnA Assistant"


# ============================================================
# PDF / RAG
# ============================================================

# Drop your knowledge PDF here.  Edit the filename if you rename it.
PDF_PATH = os.environ.get(
    "QNA_PDF_PATH",
    os.path.join(os.path.dirname(__file__), "data", "knowledge.pdf"),
)

# Cache the FAISS index so we don't re-embed every restart
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "faiss_index")

# Chunking
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 120     # overlap between chunks

# Retrieval
TOP_K = 5               # how many chunks to retrieve per question

# OCR — set to True to OCR pages that have no extractable text
OCR_ENABLED = True
OCR_LANG = "eng+urd"    # Tesseract language packs


# ============================================================
# STT (microphone capture)
# ============================================================

ENERGY_THRESHOLD = 4000
DYNAMIC_ENERGY_THRESHOLD = True
PAUSE_THRESHOLD = 0.8
LISTEN_TIMEOUT = 10
PHRASE_TIME_LIMIT = 15

# Whisper — runs on Google Colab, exposed via ngrok
WHISPER_API_URL = os.environ.get(
    "WHISPER_API_URL",
    "https://superdelicately-subsynodical-darrel.ngrok-free.dev/transcribe",
)


# ============================================================
# EDGE TTS FALLBACK (when Gemini TTS fails)
# ============================================================

EDGE_TTS_VOICE_EN = "en-US-JennyNeural"
EDGE_TTS_VOICE_UR = "ur-PK-UzmaNeural"


# ============================================================
# UI MESSAGES
# ============================================================

WELCOME_MESSAGE = "Assalam o Alaikum! Ask me anything about the document."
GOODBYE_MESSAGE = "Thank you for chatting. Have a wonderful day!"
EXIT_COMMANDS = {"exit", "quit", "stop", "goodbye", "bye"}
