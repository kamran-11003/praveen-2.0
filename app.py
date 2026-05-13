"""Flask + Socket.IO server — PAK Center QnA Avatar (fully local, single kiosk).

Socket events  (client → server)
─────────────────────────────────
  connect          → sends greeting metadata
  disconnect       → clears session state
  request_greeting → avatar speaks welcome message
  audio_data       → raw WAV bytes captured by browser VAD
  barge_in         → user spoke while avatar was talking; stop current pipeline

Socket events  (server → client)
─────────────────────────────────
  server_profile   → avatar URL / name
  status           → UI state label
  user_text        → what the user said (for chat panel)
  speak_chunk      → {text, audio(b64), words, wtimes, wdurations, lang}
  speak_done       → all chunks for this turn have been sent
  cannot_hear      → STT returned empty
"""

import base64
import logging
import threading

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import config
import qna_handler
from stt_service import transcribe
from tts_service import generate_tts

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("app")

# ── Flask / SocketIO ──────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=10 * 1024 * 1024,   # 10 MB — enough for ~30s of 16kHz WAV
)

# sid → True while a pipeline is allowed to run; False on barge-in / disconnect
_active: dict[str, bool] = {}


# ── routes ────────────────────────────────────────────────────────────────────

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
    return jsonify({"status": "ok", "model": config.OLLAMA_MODEL})


# ── TTS helper ────────────────────────────────────────────────────────────────

def _emit_speak(sid: str, text: str, lang: str) -> None:
    """Synthesise one sentence and emit a speak_chunk event."""
    try:
        audio_bytes, words, wtimes, wdurations = generate_tts(text, lang=lang)
        socketio.emit(
            "speak_chunk",
            {
                "text":       text,
                "audio":      base64.b64encode(audio_bytes).decode("ascii"),
                "words":      words,
                "wtimes":     wtimes,
                "wdurations": wdurations,
                "lang":       lang,
            },
            room=sid,
        )
    except Exception as exc:
        log.error("[%s] TTS error: %s", sid[:8], exc)
        # Emit text-only so the chat panel still updates
        socketio.emit("speak_chunk", {"text": text, "lang": lang}, room=sid)


# ── socket events ─────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    sid = request.sid
    _active[sid] = True
    emit("server_profile", {
        "assistant_name": config.AVATAR_NAME,
        "avatar_url":     config.AVATAR_URL,
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


@socketio.on("barge_in")
def on_barge_in():
    """User spoke while avatar was talking — cancel current pipeline."""
    _active[request.sid] = False


@socketio.on("request_greeting")
def on_request_greeting():
    sid = request.sid
    threading.Thread(
        target=_greet, args=(sid,), daemon=True
    ).start()


@socketio.on("audio_data")
def on_audio_data(data):
    """Receive WAV bytes from browser VAD and run the full pipeline."""
    sid = request.sid
    _active[sid] = True
    threading.Thread(
        target=_pipeline, args=(sid, bytes(data)), daemon=True
    ).start()


# ── greeting ──────────────────────────────────────────────────────────────────

def _greet(sid: str) -> None:
    socketio.emit("status", {"state": "speaking", "message": "Speaking..."}, room=sid)
    _emit_speak(sid, config.WELCOME_MESSAGE, "en")
    socketio.emit("speak_done", room=sid)


# ── main pipeline: STT → RAG → LLM → TTS ─────────────────────────────────────

def _pipeline(sid: str, audio_bytes: bytes) -> None:
    try:
        # 1. STT
        socketio.emit("status", {"state": "processing", "message": "Transcribing..."}, room=sid)
        result = transcribe(audio_bytes)
        text = result["text"].strip()
        lang = result["language"]

        if not text:
            socketio.emit("cannot_hear", room=sid)
            return

        socketio.emit("user_text", {"text": text}, room=sid)
        log.info("[%s] STT lang=%s  text=%.80s", sid[:8], lang, text)

        # 2. Exit commands
        if text.lower() in config.EXIT_COMMANDS:
            reply_lang = qna_handler._SESSION_LANG.get(sid, lang)
            _emit_speak(sid, config.GOODBYE_MESSAGE, reply_lang)
            socketio.emit("speak_done", room=sid)
            return

        # 3. LLM + sentence-level TTS streaming
        socketio.emit("status", {"state": "thinking", "message": "Thinking..."}, room=sid)

        for sentence in qna_handler.answer_stream(text, sid=sid, detected_lang=lang):
            if not _active.get(sid, False):
                log.info("[%s] pipeline cancelled (barge-in)", sid[:8])
                return
            reply_lang = qna_handler._SESSION_LANG.get(sid, lang)
            socketio.emit("status", {"state": "speaking", "message": "Speaking..."}, room=sid)
            _emit_speak(sid, sentence, reply_lang)

        # Signal client that all chunks are done
        if _active.get(sid, True):
            socketio.emit("speak_done", room=sid)

    except Exception as exc:
        import traceback
        log.error("[%s] pipeline error: %s\n%s", sid[:8], exc, traceback.format_exc())
        socketio.emit("status", {"state": "error", "message": str(exc)}, room=sid)


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Initializing RAG (PDF → embeddings → FAISS) ...")
    try:
        qna_handler.initialize()
    except Exception as exc:
        log.error("RAG init failed: %s — server will start, answers will fail", exc)

    log.info("Starting PAK Center QnA Bot on http://%s:%s", config.HOST, config.PORT)
    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
