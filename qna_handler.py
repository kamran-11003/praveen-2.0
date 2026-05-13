"""QnA handler — Ollama/Qwen2.5 with per-session memory + sentence streaming.

Key design choices:
  - Zero temperature → deterministic, no hallucination drift.
  - Strict system prompt → LLM refuses to answer outside CONTEXT.
  - RAG threshold → if best chunk < RAG_SCORE_THRESHOLD we skip LLM entirely
    and return a canned "not found" message.
  - Sentence streaming → caller gets sentences one by one so TTS can start
    before the LLM finishes generating.
  - active_flag → caller can abort the generator mid-stream on barge-in.
  - Language always updated per turn (not locked forever).
"""

import logging
import re
from collections import defaultdict
from typing import Callable, Iterator

import ollama

import config
from rag_service import RAGService

log = logging.getLogger("qna")

# Per-session rolling message history (capped to last 16 messages = 8 turns)
_MEMORY: dict[str, list] = defaultdict(list)

# Current language per session — updated on every turn
_SESSION_LANG: dict[str, str] = {}

_RAG = RAGService()

# Sentence boundary: ends with . ! ? ۔  followed by whitespace
_SENT_END = re.compile(r'(?<=[.!?۔])\s+')

# Accepted languages — anything else is mapped to English
_VALID_LANGS = {"en", "ur"}


def initialize() -> None:
    """Build or load FAISS index at startup."""
    _RAG.build_or_load()


# ── public session API ────────────────────────────────────────────────────────

def set_language(sid: str, lang: str) -> None:
    """Set (or override) the session language for sid."""
    _SESSION_LANG[sid] = lang if lang in _VALID_LANGS else "en"


def get_language(sid: str, default: str = "en") -> str:
    """Return the current language for sid."""
    return _SESSION_LANG.get(sid, default)


# ── system prompt ─────────────────────────────────────────────────────────────

def _system_prompt(lang: str) -> str:
    if lang == "ur":
        return (
            "آپ PAK Center اسلام آباد کے سرکاری مددگار ہیں۔\n"
            "آپ کو صرف اور صرف نیچے دیے گئے CONTEXT بلاک کی معلومات استعمال کرنی ہیں۔\n"
            "اگر سوال کا جواب CONTEXT میں موجود نہ ہو، تو صرف یہ کہیں: "
            "'معذرت، یہ معلومات دستاویز میں موجود نہیں۔'\n"
            "ہرگز اپنی طرف سے کوئی فیس، دستاویز، وقت یا معلومات نہ بنائیں۔\n"
            "جواب قدرتی، مختصر (2–3 جملے) اور بول چال کی اردو میں دیں — جواب آواز میں بولا جائے گا۔\n"
            "انگریزی میں جواب ہرگز نہ دیں۔"
        )
    return (
        "You are the official assistant for PAK Center Islamabad — "
        "a one-stop government service delivery center.\n"
        "RULES (follow strictly, no exceptions):\n"
        "1. Answer ONLY using the facts in the CONTEXT block below.\n"
        "2. If the answer is not found in CONTEXT, say EXACTLY: "
        "   'Sorry, I couldn't find that information in the document.'\n"
        "3. NEVER invent, guess, or extrapolate fees, document names, processing times, "
        "   phone numbers, or any other details not explicitly stated in CONTEXT.\n"
        "4. NEVER answer in any language other than English.\n"
        "5. Keep answers conversational and brief (2–3 sentences) — "
        "   they are spoken aloud by an avatar.\n"
        "6. Do not add disclaimers, apologies, or meta-commentary."
    )


# ── not-found canned responses ─────────────────────────────────────────────────

_NOT_FOUND = {
    "en": ("I'm here to help with PAK Center Islamabad services — "
           "passport, NADRA, police, CDA, IESCO, and DCO. "
           "Could you ask a specific question about one of these?"),
    "ur": "میں PAK Center اسلام آباد کی سروسز — پاسپورٹ، نادرا، پولیس، سی ڈی اے، اسکو اور ڈی سی او — کے بارے میں مدد کر سکتا ہوں۔ کیا آپ کوئی مخصوص سوال پوچھنا چاہتے ہیں؟",
}

# Simple conversational openers — bypass RAG and respond directly
_GREETING_RE = re.compile(
    r'^\s*(hi+|hello+|hey+|\u0633\u0644\u0627\u0645|\u06c1\u06cc\u0644\u0648|\u0622\u062f\u0627\u0628|greetings?|'
    r'good\s*(morning|afternoon|evening|day)|howdy|yo+|what\'?s?\s*up|'
    r'assalam|assalamu|السلام)\b',
    re.IGNORECASE,
)
_GREETING_REPLY = {
    "en": ("Hello! Welcome to PAK Center Islamabad. "
           "I can help you with passport, NADRA, police, CDA, IESCO, and DCO services. "
           "What would you like to know?"),
    "ur": ("سلام! PAK Center اسلام آباد میں خوش آمدید۔ "
           "میں پاسپورٹ، نادرا، پولیس، سی ڈی اے، اسکو اور ڈی سی او سروسز کے بارے میں مدد کر سکتا ہوں۔ "
           "آپ کیا جاننا چاہتے ہیں؟"),
}


# ── streaming answer ──────────────────────────────────────────────────────────

def answer_stream(
    question: str,
    sid: str,
    detected_lang: str = "en",
    active_flag: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """Yield complete sentences one by one as Ollama streams tokens.

    The caller (app.py) sends each sentence to TTS immediately, so the avatar
    starts speaking the first sentence while the LLM generates the rest.

    active_flag — zero-argument callable; if it returns False the generator
    stops consuming tokens immediately (used for barge-in cancellation).
    """
    # Enforce language; update every turn so bilingual users switch seamlessly
    lang = detected_lang if detected_lang in _VALID_LANGS else "en"
    _SESSION_LANG[sid] = lang
    log.info("[%s] language → %s", sid[:8], lang)

    # Short-circuit for greetings / small talk — no RAG needed
    if _GREETING_RE.match(question):
        reply = _GREETING_REPLY.get(lang, _GREETING_REPLY["en"])
        _MEMORY[sid].append({"role": "user",      "content": question})
        _MEMORY[sid].append({"role": "assistant", "content": reply})
        _MEMORY[sid] = _MEMORY[sid][-16:]
        log.info("[%s] greeting bypass — skipping RAG", sid[:8])
        yield reply
        return

    # RAG retrieval with confidence gate
    context_text, found = _RAG.context_for(question)
    if not found:
        log.info("[%s] RAG: no confident match — returning not-found", sid[:8])
        yield _NOT_FOUND.get(lang, _NOT_FOUND["en"])
        return

    # Build message list
    messages = [{"role": "system", "content": _system_prompt(lang)}]
    messages.extend(_MEMORY[sid][-16:])   # last 8 turns (already capped below)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}",
    })

    # Stream tokens from Ollama, yield complete sentences
    buf = ""
    full_text = ""
    try:
        stream = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0.0, "num_predict": 300},
        )
        for chunk in stream:
            # Abort mid-stream on barge-in
            if active_flag is not None and not active_flag():
                log.info("[%s] answer_stream: aborted by active_flag", sid[:8])
                return

            token = chunk["message"]["content"]
            buf += token
            full_text += token

            # Emit every complete sentence immediately
            parts = _SENT_END.split(buf)
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    log.debug("[%s] sentence: %.60s", sid[:8], sentence)
                    yield sentence
            buf = parts[-1]

        # Flush any remaining text
        if buf.strip():
            yield buf.strip()

    except ollama.ResponseError as exc:
        log.error("[%s] Ollama error: %s", sid[:8], exc)
        yield _NOT_FOUND.get(lang, _NOT_FOUND["en"])
        return

    # Update session memory, capped at 16 messages (8 turns)
    if full_text.strip():
        _MEMORY[sid].append({"role": "user",      "content": question})
        _MEMORY[sid].append({"role": "assistant", "content": full_text.strip()})
        _MEMORY[sid] = _MEMORY[sid][-16:]   # prevent unbounded growth
        log.info("[%s] Q: %.60s  A: %.60s", sid[:8], question, full_text.replace("\n", " "))


def clear_session(sid: str) -> None:
    _MEMORY.pop(sid, None)
    _SESSION_LANG.pop(sid, None)
