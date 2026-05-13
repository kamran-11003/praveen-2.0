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
import json
import logging
import os
import threading

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import config
import qna_handler
from conversation_logger import ConversationLogger
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

# sid -> True while a pipeline is allowed to run; False on barge-in / disconnect
_active: dict[str, bool] = {}

# Conversation logger (daily-rotated JSONL in config.LOG_DIR)
_conv_log = ConversationLogger(config.LOG_DIR)

# Services catalog (loaded once at startup)
_CATALOG: dict = {"departments": [], "services": []}
_SERVICE_BY_ID: dict[str, dict] = {}


def _load_catalog() -> None:
    """Load services_catalog_canonical.json once at startup."""
    global _CATALOG, _SERVICE_BY_ID
    if not os.path.exists(config.CATALOG_PATH):
        log.warning("Catalog not found at %s - services panel will be empty", config.CATALOG_PATH)
        return
    try:
        with open(config.CATALOG_PATH, "r", encoding="utf-8") as f:
            _CATALOG = json.load(f)
        _SERVICE_BY_ID = {s["service_id"]: s for s in _CATALOG.get("services", [])}
        log.info(
            "Loaded catalog: %d departments, %d services",
            len(_CATALOG.get("departments", [])),
            len(_SERVICE_BY_ID),
        )
    except Exception as exc:
        log.error("Failed to load catalog: %s", exc)


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


@app.route("/api/services")
def api_services():
    """Return the full services catalog (departments + services) as JSON.
    Browser uses this to populate the services panel on the right side.
    """
    return jsonify(_CATALOG)


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
    """User spoke while avatar was talking - cancel current pipeline."""
    sid = request.sid
    _active[sid] = False
    _conv_log.log(sid, role="system", text="barge-in: user interrupted avatar",
                  lang=qna_handler._SESSION_LANG.get(sid, "en"), event="barge_in")


@socketio.on("service_selected")
def on_service_selected(data):
    """User clicked a service card on the right panel.
    data = {"serviceId": "SRV_xxx", "lang": "en" | "ur"}
    """
    sid         = request.sid
    service_id  = (data or {}).get("serviceId", "")
    lang        = (data or {}).get("lang", "en")
    if lang not in ("en", "ur"):
        lang = "en"
    if not service_id or service_id not in _SERVICE_BY_ID:
        log.warning("[%s] service_selected: unknown id '%s'", sid[:8], service_id)
        return
    _active[sid] = True
    threading.Thread(
        target=_service_pipeline, args=(sid, service_id, lang), daemon=True
    ).start()


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
    _conv_log.log(sid, role="assistant", text=config.WELCOME_MESSAGE,
                  lang="en", event="greeting")
    _emit_speak(sid, config.WELCOME_MESSAGE, "en")
    socketio.emit("speak_done", room=sid)


# -- service-card pipeline (no LLM, deterministic spoken summary) -------------

def _format_service_response(service: dict, lang: str) -> list[str]:
    """Build 2-3 short spoken sentences describing the selected service."""
    name_obj = service.get("service_name", {}) or {}
    name     = (name_obj.get(lang) or name_obj.get("en") or "this service").strip()
    summary  = (service.get("summary") or "").strip()

    pay      = service.get("payment_model", {}) or {}
    pay_type = pay.get("type", "")
    pay_entries = pay.get("entries") or []

    docs = service.get("required_documents") or []

    if lang == "ur":
        sentences = [f"آپ نے منتخب کیا: {name}۔"]
        if pay_type == "free":
            sentences.append("یہ سروس مفت فراہم کی جاتی ہے۔")
        elif pay_entries:
            sentences.append(f"فیس: {pay_entries[0]}۔")
        if docs:
            sentences.append(f"کل {len(docs)} دستاویزات درکار ہیں۔ تفصیل کے لیے پوچھیے۔")
    else:
        sentences = [f"You selected: {name}."]
        if summary:
            short = summary.split(". ")[0].strip().rstrip(".") + "."
            if len(short) > 200:
                short = short[:197].rstrip() + "..."
            sentences.append(short)
        if pay_type == "free":
            sentences.append("This service is free of charge.")
        elif pay_entries:
            sentences.append(f"Fee: {pay_entries[0]}.")
        if docs:
            sentences.append(f"You will need {len(docs)} documents. Ask me for the full list.")
    return sentences


def _service_pipeline(sid: str, service_id: str, lang: str) -> None:
    try:
        service = _SERVICE_BY_ID.get(service_id)
        if not service:
            return

        # Lock session language so any follow-up question stays in this lang
        qna_handler._SESSION_LANG[sid] = lang

        name_obj = service.get("service_name", {}) or {}
        name     = (name_obj.get(lang) or name_obj.get("en") or service_id).strip()
        _conv_log.log(sid, role="user", text=f"[card] {name}",
                      lang=lang, event="service_click")

        socketio.emit("user_text", {"text": name}, room=sid)
        socketio.emit("status", {"state": "speaking", "message": "Speaking..."}, room=sid)

        for sentence in _format_service_response(service, lang):
            if not _active.get(sid, False):
                log.info("[%s] service_pipeline cancelled (barge-in)", sid[:8])
                return
            _conv_log.log(sid, role="assistant", text=sentence,
                          lang=lang, event="service_click")
            _emit_speak(sid, sentence, lang)

        if _active.get(sid, True):
            socketio.emit("speak_done", room=sid)
    except Exception as exc:
        import traceback
        log.error("[%s] service_pipeline error: %s\n%s",
                  sid[:8], exc, traceback.format_exc())


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
        _conv_log.log(sid, role="user", text=text, lang=lang, event="speech")
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
            _conv_log.log(sid, role="assistant", text=sentence,
                          lang=reply_lang, event="speech")
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
    log.info("Loading services catalog ...")
    _load_catalog()

    log.info("Initializing RAG (PDF -> embeddings -> FAISS) ...")
    try:
        qna_handler.initialize()
    except Exception as exc:
        log.error("RAG init failed: %s - server will start, answers will fail", exc)

    log.info("Starting PAK Center QnA Bot on http://%s:%s", config.HOST, config.PORT)
    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
