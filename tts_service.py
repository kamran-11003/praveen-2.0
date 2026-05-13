"""TTS service — Microsoft Edge TTS (en-US-JennyNeural / ur-PK-UzmaNeural).

Edge TTS is free, high-quality, and returns real word-boundary events
which drive precise avatar lip-sync.

Returns: (audio_bytes, words, wtimes_ms, wdurations_ms)
"""

import asyncio
import threading

import edge_tts

import config

# Cap concurrent TTS calls to avoid Edge TTS 429 rate-limit
_TTS_SEMAPHORE = threading.Semaphore(3)


# ──────────────────────────────────────────────────────────────────────────────
# Edge TTS
# ──────────────────────────────────────────────────────────────────────────────

async def _edge_tts_async(text: str, voice: str):
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    audio_chunks: list[bytes] = []
    boundaries: list[dict] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            boundaries.append({
                "text":     chunk["text"],
                "offset":   chunk["offset"],
                "duration": chunk["duration"],
            })
    return b"".join(audio_chunks), boundaries


def generate_tts(text: str, lang: str = "en") -> tuple[bytes, list[str], list[int], list[int]]:
    """Synthesise `text` and return (audio_bytes, words, wtimes_ms, wdurations_ms)."""
    voice = config.EDGE_TTS_VOICE_UR if lang == "ur" else config.EDGE_TTS_VOICE_EN
    with _TTS_SEMAPHORE:
        audio_bytes, boundaries = asyncio.run(_edge_tts_async(text, voice))

    words      = [b["text"] for b in boundaries]
    wtimes     = [int(b["offset"]   / 10_000) for b in boundaries]   # 100ns → ms
    wdurations = [int(b["duration"] / 10_000) for b in boundaries]
    return audio_bytes, words, wtimes, wdurations
