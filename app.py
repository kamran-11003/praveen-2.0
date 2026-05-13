"""Flask + Socket.IO server — PAK Center QnA Avatar (fully local, single kiosk).

Socket events  (client → server)
─────────────────────────────────
  connect            → sends greeting metadata
  disconnect         → clears session state
  request_greeting   → avatar speaks welcome message
  audio_data         → raw WAV bytes captured by browser VAD
  barge_in           → user spoke while avatar was talking; stop current pipeline
  service_selected   → user clicked a service card; avatar speaks summary
  service_navigated  → client navigated wizard to a service page (panel→voice sync)
  reset_session      → clear memory and restart
  set_language       → store preferred language

Socket events  (server → client)
─────────────────────────────────
  server_profile   → avatar URL / name
  status           → UI state label
  user_text        → what the user said (for chat panel)
  speak_chunk      → {text, audio(b64), words, wtimes, wdurations, lang}
  speak_done       → all chunks for this turn have been sent
  cannot_hear      → STT returned empty
  navigate_panel   → {serviceId} tell client to show a service detail card
  session_end      → server decided the session is complete; client resets wizard
"""

import base64
import json
import logging
import os
import re
import threading

# Disable TensorFlow inside transformers — TF 2.10 is incompatible with NumPy 2.x
os.environ["USE_TF"] = "0"

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
_ALLOWED_ORIGINS = [
    "http://localhost:5005",
    "http://127.0.0.1:5005",
    f"http://localhost:{config.PORT}",
    f"http://127.0.0.1:{config.PORT}",
]
# Allow extra origins (e.g. ngrok tunnel) via env var:
#   set EXTRA_ORIGINS=https://xxxx.ngrok-free.dev
_extra = os.environ.get("EXTRA_ORIGINS", "")
if _extra:
    _ALLOWED_ORIGINS.extend([o.strip() for o in _extra.split(",") if o.strip()])
CORS(app, origins=_ALLOWED_ORIGINS)
socketio = SocketIO(
    app,
    cors_allowed_origins=_ALLOWED_ORIGINS,
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


# ── Service keyword search index for panel navigation ─────────────────────────
# Manual phrase → service_id mapping (covers most spoken queries)
_MANUAL_PHRASES: dict[str, list[str]] = {
    "SRV_021": ["new passport", "fresh passport", "apply for passport", "first passport",
                "passport application", "apply passport", "get a passport",
                "نیا پاسپورٹ", "پاسپورٹ بنوانا", "پاسپورٹ بنانا", "پہلا پاسپورٹ"],
    "SRV_023": ["renew passport", "passport renewal", "renew my passport", "renew the passport",
                "passport expired", "expired passport", "passport expir", "renewal of passport",
                "passport renew",
                "پاسپورٹ رینیو", "پاسپورٹ کی تجدید", "پاسپورٹ رینیو کروانا", "پاسپورٹ تجدید",
                "پاسپورٹ میعاد", "پاسپورٹ ختم"],
    "SRV_025": ["passport modification", "modify passport", "change passport", "passport correction",
                "correct passport", "update passport",
                "پاسپورٹ تبدیل", "پاسپورٹ درست", "پاسپورٹ میں ترمیم"],
    "SRV_027": ["lost passport", "stolen passport", "missing passport", "passport lost",
                "passport stolen", "passport missing",
                "پاسپورٹ گم", "پاسپورٹ چوری", "گم شدہ پاسپورٹ"],
    "SRV_029": ["exhausted passport", "passport full", "pages finished", "all pages used",
                "no more pages in passport",
                "پاسپورٹ بھر گیا", "صفحات ختم"],
    "SRV_031": ["damaged passport", "torn passport", "ruined passport", "broken passport",
                "پاسپورٹ خراب", "پاسپورٹ پھٹا"],
    "SRV_037": ["renew driving", "driving license renewal", "driving licence renewal",
                "license renewal", "renew license", "renew licence", "driving license", "driving licence",
                "ڈرائیونگ لائسنس رینیو", "ڈرائیونگ لائسنس نیو", "ڈرائیونگ لائسنس تجدید",
                "ڈرائیونگ لائسنس", "لائسنس رینیو", "لائسنس بنوانا", "لائسنس بنانا", "لائسنس"],
    "SRV_038": ["duplicate driving", "lost driving license", "lost driving licence",
                "ڈپلیکیٹ لائسنس", "لائسنس گم", "گم شدہ لائسنس"],
    "SRV_039": ["learner permit", "learning license", "learning licence", "learner license",
                "لرنر پرمٹ", "سیکھنے کا لائسنس", "نیا لائسنس سیکھنا"],
    "SRV_011": ["electricity bill", "iesco bill", "light bill", "bijli bill", "pay bill",
                "bill payment",
                "بجلی کا بل", "بجلی بل", "اسکو بل", "لائٹ بل", "بجلی ادائیگی"],
    "SRV_012": ["new electricity connection", "new meter", "new connection electricity",
                "electricity connection",
                "نیا میٹر", "نیا بجلی کنکشن", "بجلی کنکشن"],
    "SRV_014": ["electricity installment", "bill installment", "installment plan electricity",
                "بل قسط", "بجلی قسط", "قسطوں میں بل"],
    "SRV_017": ["wrong reading", "meter reading wrong", "incorrect reading",
                "غلط ریڈنگ", "میٹر ریڈنگ غلط"],
    "SRV_035": ["police verification", "verification certificate", "police verif",
                "پولیس تصدیق", "تصدیقی سرٹیفکیٹ", "پولیس ویریفکیشن"],
    "SRV_036": ["character certificate", "good conduct certificate", "character cert",
                "کردار سرٹیفکیٹ", "اچھے کردار کا سرٹیفکیٹ"],
    "SRV_040": ["tenant registration", "tenant register", "register tenant",
                "کرایہ دار رجسٹریشن", "کرایہ دار رجسٹر"],
    "SRV_041": ["missing report", "lost report", "fir report", "report missing",
                "گمشدگی رپورٹ", "ایف آئی آر", "مسنگ رپورٹ"],
    "SRV_045": ["copy of fir", "fir copy",
                "ایف آئی آر کاپی", "ایف آئی آر نقل"],
    "SRV_046": ["crime report", "incident report",
                "جرم رپورٹ", "واقعہ رپورٹ"],
    "SRV_047": ["e-challan", "challan review", "traffic challan",
                "ای چالان", "ٹریفک چالان", "چالان"],
    "SRV_049": ["building plan", "construction plan", "building approval", "approve plan",
                "تعمیراتی منصوبہ", "بلڈنگ پلان", "تعمیر کی منظوری"],
    "SRV_050": ["completion certificate", "building complete",
                "تکمیل سرٹیفکیٹ", "عمارت مکمل"],
    "SRV_051": ["noc transfer", "transfer noc", "property transfer",
                "جائیداد منتقلی", "این او سی ٹرانسفر"],
    "SRV_052": ["lease extension", "noc lease",
                "لیز توسیع", "لیز این او سی"],
    "SRV_053": ["demolition permission", "demolish building",
                "گراؤ اجازت", "عمارت گرانا"],
    "SRV_054": ["death certificate", "death registration", "register death",
                "وفات سرٹیفکیٹ", "موت سرٹیفکیٹ", "وفات رجسٹریشن"],
    "SRV_058": ["birth certificate", "birth registration", "register birth",
                "پیدائش سرٹیفکیٹ", "پیدائش رجسٹریشن", "برتھ سرٹیفکیٹ"],
    "SRV_063": ["id card", "cnic", "national identity card", "identity card", "nadra card",
                "new cnic", "renew cnic", "duplicate cnic",
                "شناختی کارڈ", "سی این آئی سی", "نادرا کارڈ", "قومی شناختی کارڈ",
                "سی این آئی سی رینیو", "کارڈ رینیو"],
    "SRV_066": ["family registration certificate", "family certificate", "frc",
                "family reg",
                "خاندانی رجسٹریشن", "فیملی سرٹیفکیٹ", "ایف آر سی"],
    "SRV_067": ["child registration certificate", "child certificate", "crc",
                "child reg",
                "بچے کا سرٹیفکیٹ", "بچہ رجسٹریشن", "سی آر سی"],
    "SRV_069": ["pakistan origin card", "poc card", "overseas identity", "origin card",
                "پاکستان اوریجن کارڈ", "پی او سی", "بیرون ملک شناخت"],
    "SRV_072": ["biometric verification", "biometric",
                "بایومیٹرک", "انگلیوں کے نشانات"],
    "SRV_074": ["domicile certificate", "domicile",
                "ڈومیسائل", "ڈومیسائل سرٹیفکیٹ", "رہائشی سرٹیفکیٹ"],
    "SRV_076": ["divorce certificate", "divorce registration", "divorce cert",
                "طلاق سرٹیفکیٹ", "طلاق رجسٹریشن"],
    "SRV_077": ["marriage certificate", "marriage registration", "nikah certificate",
                "marriage cert",
                "نکاح سرٹیفکیٹ", "شادی سرٹیفکیٹ", "نکاح نامہ", "شادی رجسٹریشن"],
    "SRV_079": ["token tax", "vehicle tax", "motor vehicle tax", "car tax",
                "گاڑی ٹیکس", "ٹوکن ٹیکس", "موٹر ویہیکل ٹیکس"],
    "SRV_006": ["document attestation", "general attestation", "attest document", "attestation",
                "دستاویز تصدیق", "اٹیسٹیشن"],
    "SRV_004": ["apostille", "اپوسٹیل"],
}

_SERVICE_SEARCH_INDEX: list[tuple[str, str]] = []   # sorted (phrase, svc_id) longest first


def _build_service_search_index() -> None:
    """Build phrase → service_id lookup from manual phrases + catalog names."""
    global _SERVICE_SEARCH_INDEX
    entries: list[tuple[str, str]] = []

    # 1. Manual phrases (most reliable)
    for svc_id, phrases in _MANUAL_PHRASES.items():
        for phrase in phrases:
            entries.append((phrase.lower(), svc_id))

    # 2. Auto-add from catalog service names (fallback)
    for svc_id, svc in _SERVICE_BY_ID.items():
        name = (svc.get("service_name", {}) or {}).get("en", "")
        clean = re.sub(r'\s*\([^)]*\)', '', name).lower().strip()
        if clean and len(clean) > 5:
            entries.append((clean, svc_id))
        # Also Urdu name
        name_ur = (svc.get("service_name", {}) or {}).get("ur", "")
        if name_ur and name_ur.strip():
            entries.append((name_ur.strip(), svc_id))

    # Sort longest-phrase-first so more specific matches win
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    _SERVICE_SEARCH_INDEX = entries
    log.info("Service search index: %d entries", len(entries))


def _match_service(text: str) -> str | None:
    """Return service_id if the text appears to be about a known service."""
    t = text.lower()
    for phrase, svc_id in _SERVICE_SEARCH_INDEX:
        if phrase in t:
            log.info("Service match: '%s' → %s", phrase, svc_id)
            return svc_id
    return None


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
    """User spoke while avatar was talking - cancel current pipeline then re-arm."""
    sid = request.sid
    _active[sid] = False          # stop the running pipeline immediately
    _conv_log.log(sid, role="system", text="barge-in: user interrupted avatar",
                  lang=qna_handler.get_language(sid), event="barge_in")
    # Re-arm so the next audio_data event can start a fresh pipeline
    def _rearm():
        import time; time.sleep(0.15)
        _active[sid] = True
    threading.Thread(target=_rearm, daemon=True).start()


@socketio.on("reset_session")
def on_reset_session():
    """Client requested a new session — clear conversation memory."""
    sid = request.sid
    _active[sid] = True
    qna_handler.clear_session(sid)
    _conv_log.log(sid, role="system", text="session reset by user",
                  lang="en", event="reset")
    log.info("session reset: %s", sid[:8])


@socketio.on("set_language")
def on_set_language(data):
    """Wizard Step 1 language selection: store preferred lang for this session."""
    sid  = request.sid
    lang = (data or {}).get("lang", "en")
    qna_handler.set_language(sid, lang)
    log.info("[%s] language set to %s via wizard", sid[:8], lang)


@socketio.on("service_navigated")
def on_service_navigated(data):
    """Client wizard navigated to a service detail page — speak its summary."""
    sid        = request.sid
    service_id = (data or {}).get("serviceId", "")
    lang       = qna_handler.get_language(sid, "en")
    if not service_id or service_id not in _SERVICE_BY_ID:
        return
    _active[sid] = True
    threading.Thread(
        target=_service_pipeline, args=(sid, service_id, lang), daemon=True
    ).start()


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
        qna_handler.set_language(sid, lang)
        # Tell client to show this service detail in the right panel
        socketio.emit("navigate_panel", {"serviceId": service_id}, room=sid)

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
            # Ask if user needs anything else after speaking the service summary
            followup = ("Is there anything else I can help you with?"
                        if lang == "en"
                        else "کیا میں آپ کی مزید مدد کر سکتا ہوں؟")
            _emit_speak(sid, followup, lang)
            socketio.emit("speak_done", {"expectReply": True}, room=sid)
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

        # 2. Exit / satisfaction commands — if user says goodbye/no more, end session
        _DONE_EN = {"no","no thanks","no thank you","nope","that's all","that is all",
                    "i'm good","i am good","goodbye","bye","thanks","thank you","exit","quit"}
        _DONE_UR = {"نہیں","شکریہ","خدا حافظ","بس","کافی ہے"}
        text_lower = text.lower().strip().rstrip('.')
        reply_lang = qna_handler.get_language(sid, lang)
        if text_lower in config.EXIT_COMMANDS or text_lower in _DONE_EN or text in _DONE_UR:
            bye = (config.GOODBYE_MESSAGE if text_lower in config.EXIT_COMMANDS
                   else ("Thank you for visiting PAK Center. Have a great day!"
                         if reply_lang == "en"
                         else "PAK Center آنے کا شکریہ۔ آپ کا دن اچھا گزرے!"))
            _emit_speak(sid, bye, reply_lang)
            socketio.emit("speak_done", {"resetSession": True}, room=sid)
            qna_handler.clear_session(sid)
            return

        # 3. Service keyword match → navigate panel card (still run LLM for the actual answer)
        svc_id = _match_service(text)
        if svc_id:
            socketio.emit("navigate_panel", {"serviceId": svc_id}, room=sid)
            # Do NOT return — fall through to LLM so the specific question is answered

        # 4. LLM + sentence-level TTS streaming
        socketio.emit("status", {"state": "thinking", "message": "Thinking..."}, room=sid)

        # Collect all sentences and check the full text for wrap-up detection
        sentences_out = []
        for sentence in qna_handler.answer_stream(
            text, sid=sid, detected_lang=lang,
            active_flag=lambda: _active.get(sid, False),
        ):
            if not _active.get(sid, False):
                log.info("[%s] pipeline cancelled (barge-in)", sid[:8])
                return
            reply_lang = qna_handler.get_language(sid, lang)
            _conv_log.log(sid, role="assistant", text=sentence,
                          lang=reply_lang, event="speech")
            socketio.emit("status", {"state": "speaking", "message": "Speaking..."}, room=sid)
            _emit_speak(sid, sentence, reply_lang)
            sentences_out.append(sentence)

        # After answering, ask if user needs anything else
        if _active.get(sid, True) and sentences_out:
            reply_lang = qna_handler.get_language(sid, lang)
            followup = ("Is there anything else I can help you with?"
                        if reply_lang == "en"
                        else "کیا میں آپ کی مزید مدد کر سکتا ہوں؟")
            _emit_speak(sid, followup, reply_lang)

        # Signal client that all chunks are done
        if _active.get(sid, True):
            socketio.emit("speak_done", {"expectReply": True}, room=sid)

    except Exception as exc:
        import traceback
        log.error("[%s] pipeline error: %s\n%s", sid[:8], exc, traceback.format_exc())
        socketio.emit("status", {"state": "error", "message": str(exc)}, room=sid)


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Loading services catalog ...")
    _load_catalog()
    _build_service_search_index()

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
