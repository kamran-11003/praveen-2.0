"""TTS service: Gemini native TTS first, Edge TTS fallback.

Both backends return:
    (mp3_bytes, words, wtimes_ms, wdurations_ms)

Edge TTS provides real word-boundary timestamps (great lip-sync).
Gemini TTS does not — we estimate evenly-spaced word timings from audio duration.
"""

import asyncio
import io
import wave

import edge_tts

import config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container so the browser can decode it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)        # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def _estimate_word_timings(text: str, duration_ms: int) -> tuple[list[str], list[int], list[int]]:
    """Distribute words evenly across `duration_ms` (no real boundaries from Gemini TTS)."""
    words = [w for w in text.split() if w]
    if not words:
        return [], [], []
    per = max(1, duration_ms // len(words))
    wtimes = [i * per for i in range(len(words))]
    wdurations = [per] * len(words)
    return words, wtimes, wdurations


# ──────────────────────────────────────────────────────────────────────────────
# Gemini TTS (preview)
# ──────────────────────────────────────────────────────────────────────────────

def _gemini_tts(text: str, lang: str) -> tuple[bytes, list[str], list[int], list[int]] | None:
    """Try Gemini's native TTS.  Returns None on any failure."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        voice = config.GEMINI_TTS_VOICE_UR if lang == "ur" else config.GEMINI_TTS_VOICE_EN

        response = client.models.generate_content(
            model=config.GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice),
                    ),
                ),
            ),
        )

        # PCM 24kHz mono returned as base64-decoded bytes in inline_data.data
        part = response.candidates[0].content.parts[0]
        pcm_bytes = part.inline_data.data
        wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate=24000)

        # Estimate timings from duration (44 + N samples, 24000 Hz, 16-bit mono)
        sample_count = max(1, (len(wav_bytes) - 44) // 2)
        duration_ms = int((sample_count / 24000) * 1000)
        words, wtimes, wdurations = _estimate_word_timings(text, duration_ms)

        return wav_bytes, words, wtimes, wdurations

    except Exception as exc:
        print(f"[tts] Gemini TTS failed: {exc} — falling back to Edge TTS")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Edge TTS (fallback, real word boundaries)
# ──────────────────────────────────────────────────────────────────────────────

async def _edge_tts_async(text: str, voice: str):
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    audio_chunks: list[bytes] = []
    boundaries = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            boundaries.append({
                "text": chunk["text"],
                "offset": chunk["offset"],
                "duration": chunk["duration"],
            })
    return b"".join(audio_chunks), boundaries


def _edge_tts(text: str, lang: str) -> tuple[bytes, list[str], list[int], list[int]]:
    voice = config.EDGE_TTS_VOICE_UR if lang == "ur" else config.EDGE_TTS_VOICE_EN
    loop = asyncio.new_event_loop()
    try:
        audio_bytes, boundaries = loop.run_until_complete(_edge_tts_async(text, voice))
    finally:
        loop.close()

    words      = [b["text"] for b in boundaries]
    wtimes     = [int(b["offset"] / 10000) for b in boundaries]
    wdurations = [int(b["duration"] / 10000) for b in boundaries]
    return audio_bytes, words, wtimes, wdurations


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_tts(text: str, lang: str = "en") -> tuple[bytes, list[str], list[int], list[int]]:
    """Try Gemini TTS first; fall back to Edge TTS on any failure."""
    result = _gemini_tts(text, lang)
    if result is not None:
        return result
    return _edge_tts(text, lang)
