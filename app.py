"""Flask + Socket.IO server for the PDF QnA Avatar Bot.

Endpoints
---------
GET  /        → renders the avatar UI (templates/avatar.html)
GET  /health  → health check

Socket events (server side)
---------------------------
connect             → sends greeting metadata
disconnect          → clears session memory + language lock
request_greeting    → speaks the welcome line
start_listening     → spawns mic-capture → STT → RAG → TTS pipeline
stop_listening      → user pressed Stop
"""

import base64
import logging
import threading

import speech_recognition as sr
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import config
import qna_handler
from stt_service import STTService
from tts_service import generate_tts


# ──────────────────────────────────────────────────────────────────────────────
# App / runtime
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

stt_service = STTService()
recognizer = sr.Recognizer()
recognizer.energy_threshold = config.ENERGY_THRESHOLD
recognizer.dynamic_energy_threshold = config.DYNAMIC_ENERGY_THRESHOLD
recognizer.pause_threshold = config.PAUSE_THRESHOLD

_active: dict[str, bool] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "avatar.html",
        avatar_url=config.AVATAR_URL,
        avatar_fallback=config.AVATAR_FALLBACK,
        avatar_name=config.AVATAR_NAME,
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": config.GEMINI_MODEL})


# ──────────────────────────────────────────────────────────────────────────────
# TTS helper
# ──────────────────────────────────────────────────────────────────────────────

def _emit_assistant_speak(sid: str, text: str, lang: str):
    try:
        audio_bytes, words, wtimes, wdurations = generate_tts(text, lang=lang)
        socketio.emit(
            "assistant_speak",
            {
                "text": text,
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
                "words": words,
                "wtimes": wtimes,
                "wdurations": wdurations,
                "lang": lang,
            },
            room=sid,
        )
    except Exception as exc:
        log.error("TTS failed: %s", exc)
        socketio.emit("assistant_speak", {"text": text, "lang": lang}, room=sid)


# ──────────────────────────────────────────────────────────────────────────────
# Socket events
# ──────────────────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    sid = request.sid
    _active[sid] = True
    emit("server_profile", {
        "assistant_name": config.AVATAR_NAME,
        "avatar_url": config.AVATAR_URL,
        "avatar_fallback": config.AVATAR_FALLBACK,
    })
    emit("status", {"state": "connected", "message": "Connected"})
    log.info("connected: %s", sid)


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    _active.pop(sid, None)
    qna_handler.clear_session(sid)
    log.info("disconnected: %s", sid)


@socketio.on("stop_listening")
def on_stop_listening():
    _active[request.sid] = False


@socketio.on("request_greeting")
def on_request_greeting():
    _emit_assistant_speak(request.sid, config.WELCOME_MESSAGE, "en")


@socketio.on("start_listening")
def on_start_listening():
    sid = request.sid
    _active[sid] = True
    threading.Thread(target=_listen_and_respond, args=(sid,), daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# Mic → STT → RAG → TTS
# ──────────────────────────────────────────────────────────────────────────────

def _listen_and_respond(sid: str):
    try:
        mic = sr.Microphone()
    except Exception as exc:
        socketio.emit("status", {"state": "error", "message": f"Microphone: {exc}"}, room=sid)
        return

    try:
        with mic as source:
            socketio.emit("status", {"state": "listening", "message": "Listening..."}, room=sid)
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source,
                timeout=config.LISTEN_TIMEOUT,
                phrase_time_limit=config.PHRASE_TIME_LIMIT,
            )

        socketio.emit("status", {"state": "processing", "message": "Transcribing..."}, room=sid)
        result = stt_service.transcribe(audio.get_wav_data())
        text_en = (result.get("text") or "").strip()
        text_ur = (result.get("text_ur") or text_en).strip()
        lang = result.get("language", "en")
        log.info("[%s] STT lang=%s  en=%.80s", sid[:8], lang, text_en)

        if not text_en:
            socketio.emit("cannot_hear", room=sid)
            return

        # Show user's words in their own script
        socketio.emit("user_text", {"text": text_ur if lang == "ur" else text_en}, room=sid)

        if text_en.lower() in config.EXIT_COMMANDS:
            _emit_assistant_speak(sid, config.GOODBYE_MESSAGE, lang)
            return

        socketio.emit("status", {"state": "thinking", "message": "Thinking..."}, room=sid)
        answer = qna_handler.answer(text_en, sid=sid, detected_lang=lang)

        # Use the language locked on this session for the whole reply
        from qna_handler import _SESSION_LANG
        reply_lang = _SESSION_LANG.get(sid, lang)
        _emit_assistant_speak(sid, answer, reply_lang)

    except sr.WaitTimeoutError:
        socketio.emit("cannot_hear", room=sid)
    except Exception as exc:
        import traceback
        log.error("listen_and_respond error: %s\n%s", exc, traceback.format_exc())
        socketio.emit("status", {"state": "error", "message": str(exc)}, room=sid)


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Initializing RAG (loading PDF + building/loading FAISS index) ...")
    try:
        qna_handler.initialize()
    except Exception as exc:
        log.error("RAG init failed: %s — server will start, but answers will fail until fixed", exc)

    log.info("Starting QnA Bot on http://%s:%d", config.HOST, config.PORT)
    socketio.run(app, host=config.HOST, port=config.PORT, debug=False, allow_unsafe_werkzeug=True)
