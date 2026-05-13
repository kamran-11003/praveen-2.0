"""STT service: FFmpeg preprocessing -> faster-whisper -> Roman Urdu transliteration.

Pipeline
--------
  1. FFmpeg        : denoise, normalise, resample to 16 kHz mono WAV
                     (kiosk mics often have hum / reverb; FFmpeg cleans it before Whisper)
  2. Whisper       : local GPU transcription, auto language-detect
  3. Transliterate : if Whisper output is Roman Urdu (Latin script) -> Arabic-script Urdu
                     Pakistani speakers naturally say 'mujhe passport chahiye' which
                     Whisper often transcribes in Latin.  We convert it so the LLM
                     and vector embeddings receive proper Arabic-script Urdu.

Returns:
    {"text": str, "language": "en" | "ur"}
"""

import logging
import os
import subprocess
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


# Step 1: FFmpeg preprocessing -----------------------------------------------

def _ffmpeg_preprocess(input_path: str) -> str:
    """Run FFmpeg audio cleanup. Returns path to cleaned WAV (caller deletes).
    Falls back to input_path if FFmpeg is unavailable or returns an error.
    """
    out_fd, out_path = tempfile.mkstemp(suffix="_clean.wav")
    os.close(out_fd)

    cmd = [
        config.FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-af", (
            "highpass=f=80,"        # cut rumble below 80 Hz
            "lowpass=f=8000,"       # cut hiss above 8 kHz
            "afftdn=nf=-25,"        # FFT-based noise suppression
            "dynaudnorm=p=0.9"      # dynamic level normalisation
        ),
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        out_path,
        "-loglevel", "error",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            log.warning(
                "FFmpeg error (rc=%d): %s",
                result.returncode,
                result.stderr.decode(errors="replace"),
            )
            try: os.unlink(out_path)
            except OSError: pass
            return input_path
        log.debug("FFmpeg OK -> %s", out_path)
        return out_path
    except FileNotFoundError:
        log.warning(
            "FFmpeg not found at '%s' - skipping preprocessing.  "
            "Install: winget install --id Gyan.FFmpeg",
            config.FFMPEG_PATH,
        )
        try: os.unlink(out_path)
        except OSError: pass
        return input_path
    except subprocess.TimeoutExpired:
        log.warning("FFmpeg timed out - skipping preprocessing")
        try: os.unlink(out_path)
        except OSError: pass
        return input_path


# Step 3: Roman Urdu -> Arabic-script transliteration ------------------------

def _is_roman_urdu(text: str) -> bool:
    """True if text is mostly Latin characters (Roman Urdu style).
    Requires BOTH: >50% of alpha chars are Latin AND <25% are Arabic-script.
    This prevents Devanagari/Hindi text from being misidentified as Roman Urdu.
    """
    if not text:
        return False
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    latin  = sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    alpha  = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False
    # Must be mostly Latin AND have very few Arabic-script chars
    return (latin / alpha) > 0.5 and (arabic / alpha) < 0.25


def _transliterate_to_urdu(text: str) -> str:
    """Convert Roman Urdu (Latin) -> Arabic-script Urdu via UrduHack."""
    try:
        from urduhack.transliteration import roman_to_urdu  # noqa: PLC0415
        result = str(roman_to_urdu(text))
        log.debug("transliterate: '%s' -> '%s'", text[:60], result[:60])
        return result
    except ImportError:
        log.warning(
            "urduhack not installed - Roman Urdu will NOT be transliterated. "
            "Fix: pip install urduhack"
        )
        return text
    except Exception as exc:
        log.warning("transliteration error: %s - returning original text", exc)
        return text


# Public API -----------------------------------------------------------------

def transcribe(audio_bytes: bytes) -> dict:
    """Full pipeline: FFmpeg -> Whisper -> transliterate (if Roman Urdu).

    Returns {"text": str, "language": "en" | "ur"}
    """
    model = _load()

    raw_fd, raw_path = tempfile.mkstemp(suffix=".wav")
    os.close(raw_fd)
    with open(raw_path, "wb") as f:
        f.write(audio_bytes)

    clean_path = raw_path
    try:
        if config.FFMPEG_ENABLED:
            clean_path = _ffmpeg_preprocess(raw_path)

        segments, info = model.transcribe(
            clean_path,
            beam_size=config.WHISPER_BEAM_SIZE,
            language=None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": config.WHISPER_MIN_SILENCE_MS},
            condition_on_previous_text=False,
            temperature=0,
        )
        text = " ".join(s.text for s in segments).strip()
        lang = info.language

        # This kiosk only supports English and Urdu.
        # If Whisper detects anything other than English (Hindi, Punjabi, French,
        # etc.) it is almost certainly a Pakistani user speaking Urdu — so force
        # re-transcription with language="ur" to get proper Arabic-script output.
        if lang != "en":
            log.info(
                "STT: Whisper detected '%s' (not English) — re-transcribing forced to Urdu", lang
            )
            segments_ur, _ = model.transcribe(
                clean_path,
                beam_size=config.WHISPER_BEAM_SIZE,
                language="ur",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": config.WHISPER_MIN_SILENCE_MS},
                condition_on_previous_text=False,
                temperature=0,
            )
            text = " ".join(s.text for s in segments_ur).strip()
            lang = "ur"
            log.info("STT: forced-Urdu text: %.80s", text)

        if config.TRANSLITERATE_ROMAN_URDU and _is_roman_urdu(text):
            log.info(
                "STT: Roman Urdu detected (whisper_lang=%s) - transliterating", lang
            )
            text = _transliterate_to_urdu(text)
            lang = "ur"

        # Enforce only English or Urdu
        if lang not in ("en", "ur"):
            log.info(
                "STT: unsupported language '%s' (text=%.40s) — mapping to 'en'", lang, text
            )
            lang = "en"

        log.info("STT lang=%s  text=%.80s", lang, text)
        return {"text": text, "language": lang}

    finally:
        for path in {raw_path, clean_path}:
            try:
                os.unlink(path)
            except OSError:
                pass

