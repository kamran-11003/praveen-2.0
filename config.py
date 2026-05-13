"""Configuration — PAK Center QnA Kiosk (fully local, no cloud APIs).

Stack:
  STT  : faster-whisper (local GPU)
  LLM  : Ollama + Qwen2.5  (local GPU)
  Embed: sentence-transformers multilingual (local CPU)
  TTS  : Edge TTS  (en-US-JennyNeural / ur-PK-UzmaNeural)
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
# LLM — Ollama  (ollama serve must be running)
# Pull model first:  ollama pull qwen2.5:7b
# For 6 GB VRAM use: ollama pull qwen2.5:3b
# ============================================================

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# ============================================================
# STT — faster-whisper  (runs on GPU alongside Ollama)
# "medium" uses ~750 MB VRAM with int8; "large-v3" uses ~1.5 GB.
# If you get OOM errors, switch to "small" or "base".
# ============================================================

WHISPER_MODEL   = os.environ.get("WHISPER_MODEL",   "large-v3")
WHISPER_DEVICE  = os.environ.get("WHISPER_DEVICE",  "cuda")   # "cpu" if no GPU
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")   # int8 saves VRAM

# ============================================================
# EMBEDDINGS — sentence-transformers  (CPU, downloaded on first run ~420 MB)
# paraphrase-multilingual-MiniLM-L12-v2 handles English + Urdu in one model.
# ============================================================

EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")

# Minimum cosine similarity to accept a retrieved chunk (0–1).
# Below this threshold we tell the user the answer is not in the document.
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.30"))

# ============================================================
# AVATAR
# ============================================================

AVATAR_URL = (
    "https://raw.githubusercontent.com/met4citizen/TalkingHead/refs/heads/main/avatars/brunette.glb"
)
AVATAR_FALLBACK = (
    "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@1.7/avatars/brunette.glb"
)
AVATAR_NAME = "PAK Center Assistant"

# ============================================================
# PDF / RAG
# ============================================================

PDF_PATH = os.environ.get(
    "QNA_PDF_PATH",
    os.path.join(os.path.dirname(__file__), "data", "knowledge.pdf"),
)

# FAISS index cache — delete this folder to force a rebuild after changing the PDF
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "faiss_index")

CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 120   # overlap between chunks
TOP_K         = 5     # chunks to retrieve per query

OCR_ENABLED = True
OCR_LANG    = "eng+urd"   # Tesseract language packs

# ============================================================
# TTS — Microsoft Edge TTS  (free, high quality, no API key)
# ============================================================

EDGE_TTS_VOICE_EN = "en-US-JennyNeural"
EDGE_TTS_VOICE_UR = "ur-PK-UzmaNeural"

# ============================================================
# UI MESSAGES
# ============================================================

WELCOME_MESSAGE = (
    "Assalam-o-Alaikum! I am your PAK Center assistant. "
    "You can ask me about passport, CNIC, police verification, electricity bills, "
    "and all other services available here. How can I help you?"
)
GOODBYE_MESSAGE = "Thank you for visiting PAK Center. Have a wonderful day!"
EXIT_COMMANDS   = {"exit", "quit", "goodbye", "bye", "khuda hafiz", "allah hafiz"}
