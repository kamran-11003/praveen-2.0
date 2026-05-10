"""Speech-to-text: Colab-hosted Whisper API + Gemini transliteration.

Pipeline:
  1. POST WAV bytes to the Colab ngrok endpoint → raw text + detected language
  2. Gemini transliterates to clean English + Urdu Arabic-script

Produces:
  text     — English (for the LLM)
  text_ur  — Urdu Arabic-script (for UI + TTS)
  language — "en" | "ur"  (for session lock + voice)
"""

import json
import re

import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import config


# ──────────────────────────────────────────────────────────────────────────────
# Gemini transliteration
# ──────────────────────────────────────────────────────────────────────────────

_TRANSLITERATION_PROMPT = (
    "You are a transliterator for a Pakistani QnA kiosk.  "
    "The input is raw Whisper output: Urdu (Arabic), English, Hindi (Devanagari), "
    "Roman Urdu, regional dialect, or a mix.\n\n"
    "Return ONLY a single-line JSON object:\n"
    '{"en": "<clean English translation>", '
    '"ur": "<Urdu Arabic-script, keep English words inline>", '
    '"lang": "en"|"ur"}\n\n'
    "Rules:\n"
    " - 'lang' is decided by counting Urdu/South-Asian words vs English words "
    "in the original input.\n"
    " - In 'ur', keep English words (technical terms, names) inline as-is."
)


def _transliterate(raw_text: str) -> dict:
    if not raw_text.strip():
        return {"en": "", "ur": "", "lang": "en"}
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.0,
        )
        result = llm.invoke([
            SystemMessage(content=_TRANSLITERATION_PROMPT),
            HumanMessage(content=raw_text),
        ])
        raw = re.sub(r"^```[a-z]*\n?", "", result.content.strip())
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        return {
            "en":   (parsed.get("en")   or raw_text).strip(),
            "ur":   (parsed.get("ur")   or raw_text).strip(),
            "lang": "ur" if (parsed.get("lang") or "en").lower() == "ur" else "en",
        }
    except Exception as exc:
        print(f"[stt] Gemini transliteration failed: {exc}")
        return {"en": raw_text, "ur": raw_text, "lang": "en"}


# ──────────────────────────────────────────────────────────────────────────────
# STT service
# ──────────────────────────────────────────────────────────────────────────────

class STTService:
    def transcribe(self, wav_data: bytes) -> dict:
        """POST WAV to the Colab Whisper API and return transliterated result."""
        try:
            resp = requests.post(
                config.WHISPER_API_URL,
                files={"audio": ("audio.wav", wav_data, "audio/wav")},
                timeout=60,
                headers={"ngrok-skip-browser-warning": "true"},
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = (data.get("text") or "").strip()
        except Exception as exc:
            print(f"[stt] Whisper API error: {exc}")
            return {"text": "", "text_ur": "", "language": "en"}

        translated = _transliterate(raw_text)
        return {
            "text":     translated["en"],
            "text_ur":  translated["ur"],
            "language": translated["lang"],
        }
