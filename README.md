# PDF QnA Avatar Bot

A standalone voice-driven Q&A kiosk that answers questions strictly from a single PDF.
Pipeline: **Microphone → Whisper (large-v3) → Gemini transliteration → FAISS retrieval →
Gemini chat → Gemini TTS (Edge TTS fallback) → 3D TalkingHead avatar (lip-synced).**

```
qna_bot/
├── app.py              Flask + Socket.IO server (port 5005)
├── config.py           All settings, model names, voices, paths
├── rag_service.py      PyMuPDF + OCR → chunks → FAISS + Gemini embeddings
├── stt_service.py      Whisper + ffmpeg + Gemini transliteration
├── tts_service.py      Gemini TTS (preview) → Edge TTS fallback
├── qna_handler.py      Per-session Gemini chat with PDF context + language lock
├── requirements.txt
├── data/
│   └── knowledge.pdf   ← drop your PDF here (filename configurable in config.py)
└── templates/avatar.html
```

## 1. Install Python deps

```powershell
cd qna_bot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `PyAudio` fails on Windows:
```powershell
pip install pipwin
pipwin install pyaudio
```

## 2. Install system binaries

| Binary       | Required for          | Windows install                                    |
|--------------|-----------------------|----------------------------------------------------|
| **ffmpeg**   | STT preprocessing     | `winget install Gyan.FFmpeg`                       |
| **Tesseract**| OCR for image-only PDFs | `winget install UB-Mannheim.TesseractOCR`        |

Both must be on `PATH`. For Tesseract, install the Urdu language pack as well
(installer offers it under "Additional language data").

## 3. Drop your PDF

Place your knowledge document at `data/knowledge.pdf`
(or set `QNA_PDF_PATH` env-var to a different path).

The first run extracts text (with OCR fallback for image pages), chunks it,
embeds via Gemini, and caches the FAISS index under `data/faiss_index/`.
Delete that folder to force a rebuild (e.g. when the PDF changes).

## 4. Configure

Edit `config.py` or set env vars:

| Setting              | Default                                    | Notes                          |
|----------------------|--------------------------------------------|--------------------------------|
| `GEMINI_API_KEY`     | (baked in for convenience)                 | Override via env var           |
| `GEMINI_MODEL`       | `gemini-3.1-flash-lite-preview`            | Chat / RAG model               |
| `GEMINI_TTS_MODEL`   | `gemini-2.5-flash-preview-tts`             | Falls back to Edge TTS on fail |
| `WHISPER_MODEL`      | `large-v3`                                 | Local download on first use    |
| `CHUNK_SIZE`         | 800                                        | Characters per chunk           |
| `TOP_K`              | 5                                          | Retrieved chunks per query     |
| `PORT`               | 5005                                       |                                |

## 5. Run

```powershell
python app.py
```

Then open <http://localhost:5005>, click **Start Conversation**, and ask away.

## How it works

1. **Mic capture** (`SpeechRecognition`) → 16-bit WAV.
2. **FFmpeg** loudness-normalises and resamples to 16 kHz mono.
3. **Whisper large-v3** transcribes (auto language detection).
4. **Gemini** transliterates the raw output to clean English + Arabic-script Urdu
   and decides the session language (`en` | `ur`) — locked for the whole session.
5. **FAISS** retrieves the top-K relevant chunks from your PDF.
6. **Gemini** answers strictly from those chunks, in the locked language.
7. **Gemini TTS** speaks the answer (24 kHz PCM → wrapped as WAV).
   On failure, **Edge TTS** takes over (with real word-boundary timestamps).
8. **TalkingHead.js** plays the audio and drives the avatar's mouth via the
   word-timing array.

## Notes / limitations

- **Gemini TTS** doesn't expose word boundaries, so lip-sync uses **evenly-spaced
  estimates**. Edge TTS (the fallback) gives noticeably tighter lip-sync.
- For image-only PDFs, OCR quality depends on Tesseract + your scan resolution.
  We render at 300 DPI internally.
- The session's language is **locked on the first utterance** — change it by
  refreshing the page.
- Goodbye words: `exit / quit / stop / goodbye / bye`.
