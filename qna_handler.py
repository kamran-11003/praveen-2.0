"""QnA handler — Gemini chat over PDF context + per-session language lock + memory."""

import logging
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from rag_service import RAGService

log = logging.getLogger("qna")

# Per-session rolling memory (last 8 turns)
_MEMORY: dict[str, list] = defaultdict(list)

# Session language lock (set once on first utterance, never changes)
_SESSION_LANG: dict[str, str] = {}

# Singleton RAG service
_RAG = RAGService()


def initialize() -> None:
    """Build / load the FAISS index at startup so the first query is fast."""
    _RAG.build_or_load()


def _system_prompt(session_lang: str) -> str:
    lang_label = "Urdu" if session_lang == "ur" else "English"
    script_rule = (
        "Use Urdu in Arabic/Nastaliq script (NOT Roman Urdu, NOT Devanagari). "
        "Keep technical English terms inline as-is."
        if session_lang == "ur"
        else "Use clear, simple English."
    )
    return (
        "You are a friendly assistant answering questions strictly from the provided document.\n"
        "Speak naturally and warmly — your answer is spoken aloud by a virtual avatar, "
        "so keep it short, conversational, and easy to listen to.\n\n"
        f"LANGUAGE: The user is speaking {lang_label}.  "
        f"You MUST reply ONLY in {lang_label} for this entire conversation.  "
        f"{script_rule}\n\n"
        "Rules:\n"
        " - Use ONLY the information in the CONTEXT below to answer.\n"
        " - If the context does not contain the answer, say so honestly "
        "   (e.g. \"I couldn't find that in the document.\").\n"
        " - Never invent facts.  Never cite page numbers unless the user asks.\n"
        " - Avoid bullet lists — speak in natural sentences.\n"
    )


def answer(question_en: str, sid: str, detected_lang: str = "en") -> str:
    """Answer a question using RAG + Gemini.

    Args:
        question_en: English form of the question (for retrieval + LLM).
        sid:        Session id (Socket.IO sid).
        detected_lang: STT-detected language for this turn — locks session on first call.
    """
    # Lock language on the first call of the session
    if sid not in _SESSION_LANG:
        _SESSION_LANG[sid] = detected_lang
        log.info("[%s] session language locked → %s", sid[:8], detected_lang)
    session_lang = _SESSION_LANG[sid]

    # Retrieve relevant chunks
    context = _RAG.context_for(question_en)

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.3,
    )

    history = _MEMORY[sid][-8:]
    user_block = (
        f"CONTEXT (extracted from the document):\n{context}\n\n"
        f"QUESTION:\n{question_en}"
    )

    messages = [SystemMessage(content=_system_prompt(session_lang))]
    messages.extend(history)
    messages.append(HumanMessage(content=user_block))

    result = llm.invoke(messages)
    text = result.content if isinstance(result.content, str) else str(result.content)
    log.info("[%s] Q: %.80s  →  A: %.80s", sid[:8], question_en, text.replace("\n", " "))

    # Update memory with the natural Q (not the context-padded version)
    _MEMORY[sid].append(HumanMessage(content=question_en))
    _MEMORY[sid].append(AIMessage(content=text))

    return text


def clear_session(sid: str) -> None:
    _MEMORY.pop(sid, None)
    _SESSION_LANG.pop(sid, None)
