"""STT service — faster-whisper running locally on GPU.

Whisper is multilingual: Urdu speech transcribes to Arabic-script Urdu,
English speech to English text.  No separate transliteration step needed.

Returns:
    {"text": str, "language": "en" | "ur"}
"""

import logging
import os
import tempfile

from faster_whisper import WhisperModel

import config

log = logging.getLogger("stt")

_model: WhisperModel | None = None


def _load() -> WhisperModel:
    global _model
    if _model is None:
        log.info(
            "Loading faster-whisper '%s' (%s on %s) ...",
            config.WHISPER_MODEL, config.WHISPER_COMPUTE, config.WHISPER_DEVICE,
        )
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )
        log.info("faster-whisper ready")
    return _model


def transcribe(audio_bytes: bytes) -> dict:
    """Transcribe raw audio bytes (WAV from browser).

    Returns {"text": str, "language": "en" | "ur"}
    """
    model = _load()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    try:
        segments, info = model.transcribe(
            path,
            beam_size=3,
            language=None,       # auto-detect (en / ur / etc.)
            vad_filter=True,     # built-in Silero VAD removes silence
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text for s in segments).strip()
        lang = info.language     # "en", "ur", "hi", ...
        # Map non-en / non-ur to en by default so the LLM prompt is correct
        if lang not in ("en", "ur"):
            lang = "en"
        log.info("STT lang=%s  text=%.80s", lang, text)
        return {"text": text, "language": lang}
    finally:
        os.unlink(path)
