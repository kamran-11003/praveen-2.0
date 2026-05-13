"""QnA handler — Ollama/Qwen2.5 with per-session memory + sentence streaming.

Key design choices:
  - Zero temperature → deterministic, no hallucination drift.
  - Strict system prompt → LLM refuses to answer outside CONTEXT.
  - RAG threshold → if best chunk < RAG_SCORE_THRESHOLD we skip LLM entirely
    and return a canned "not found" message.
  - Sentence streaming → caller gets sentences one by one so TTS can start
    before the LLM finishes generating.
"""

import logging
import re
from collections import defaultdict
from typing import Iterator

import ollama

import config
from rag_service import RAGService

log = logging.getLogger("qna")

# Per-session rolling message history (last 8 turns = 16 messages)
_MEMORY: dict[str, list] = defaultdict(list)

# Language locked on first utterance per session
_SESSION_LANG: dict[str, str] = {}

_RAG = RAGService()

# Sentence boundary: ends with . ! ? ۔  followed by whitespace
_SENT_END = re.compile(r'(?<=[.!?۔])\s+')


def initialize() -> None:
    """Build or load FAISS index at startup."""
    _RAG.build_or_load()


# ── system prompt ─────────────────────────────────────────────────────────────

def _system_prompt(lang: str) -> str:
    if lang == "ur":
        return (
            "آپ PAK Center اسلام آباد کے سرکاری مددگار ہیں۔ "
            "صرف اور صرف نیچے دیے گئے CONTEXT سے جواب دیں۔ "
            "اگر CONTEXT میں جواب نہ ہو تو بالکل یہی کہیں: "
            "'معذرت، یہ معلومات دستاویز میں موجود نہیں۔' "
            "جواب قدرتی، مختصر (2–3 جملے) اور بول چال کی زبان میں دیں — "
            "جواب آواز میں بولا جائے گا۔ "
            "کبھی بھی فیس، دستاویز یا وقت کے بارے میں اندازہ نہ لگائیں۔"
        )
    return (
        "You are the official assistant for PAK Center Islamabad — "
        "a one-stop government service delivery center. "
        "Answer ONLY using the facts in the CONTEXT block. "
        "If the answer is not in CONTEXT, say exactly: "
        "'Sorry, I couldn't find that information in the document.' "
        "Keep answers conversational and brief (2–3 sentences) — "
        "they are spoken aloud by an avatar. "
        "Never invent fees, document names, or processing times."
    )


# ── not-found canned responses ─────────────────────────────────────────────────

_NOT_FOUND = {
    "en": "Sorry, I couldn't find that information in the document.",
    "ur": "معذرت، یہ معلومات دستاویز میں موجود نہیں۔",
}


# ── streaming answer ──────────────────────────────────────────────────────────

def answer_stream(question: str, sid: str, detected_lang: str = "en") -> Iterator[str]:
    """Yield complete sentences one by one as Ollama streams tokens.

    The caller (app.py) sends each sentence to TTS immediately, so the avatar
    starts speaking the first sentence while the LLM generates the rest.
    """
    # Lock session language on first call
    if sid not in _SESSION_LANG:
        _SESSION_LANG[sid] = detected_lang
        log.info("[%s] language locked → %s", sid[:8], detected_lang)
    lang = _SESSION_LANG[sid]

    # RAG retrieval with confidence gate
    context_text, found = _RAG.context_for(question)
    if not found:
        log.info("[%s] RAG: no confident match — returning not-found", sid[:8])
        yield _NOT_FOUND.get(lang, _NOT_FOUND["en"])
        return

    # Build message list
    messages = [{"role": "system", "content": _system_prompt(lang)}]
    messages.extend(_MEMORY[sid][-16:])   # last 8 turns
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
            full_text = full_text  # already included

    except ollama.ResponseError as exc:
        log.error("[%s] Ollama error: %s", sid[:8], exc)
        yield _NOT_FOUND.get(lang, _NOT_FOUND["en"])
        return

    # Update session memory
    if full_text.strip():
        _MEMORY[sid].append({"role": "user",      "content": question})
        _MEMORY[sid].append({"role": "assistant", "content": full_text.strip()})
        log.info("[%s] Q: %.60s  A: %.60s", sid[:8], question, full_text.replace("\n", " "))


def clear_session(sid: str) -> None:
    _MEMORY.pop(sid, None)
    _SESSION_LANG.pop(sid, None)
