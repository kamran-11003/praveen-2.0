"""RAG service — PDF extraction + FAISS + local multilingual embeddings.

No cloud APIs.  Everything runs locally.

Flow (first run):
  1. Extract text from PDF via PyMuPDF; OCR image-only pages via Tesseract.
  2. Split into overlapping chunks.
  3. Embed with sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2).
     This model handles English AND Urdu Arabic-script in one embedding space,
     so a Urdu query retrieves English chunks without translation.
  4. Build a cosine-similarity FAISS index; cache it to disk.

Per query:
  context_for(query) → (context_text: str, found: bool)
"""

import logging
import os
import pickle
import shutil

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

import config

log = logging.getLogger("rag")

_TEXT_THRESHOLD = 40   # chars — below this, treat the page as image-only and OCR it

_INDEX_FILE  = os.path.join(config.INDEX_DIR, "index.faiss")
_CHUNKS_FILE = os.path.join(config.INDEX_DIR, "chunks.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# Text splitting  (no langchain needed)
# ──────────────────────────────────────────────────────────────────────────────

def _split(text: str, size: int, overlap: int) -> list[str]:
    """Naive recursive character splitter."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# OCR helper
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_page(page) -> str:
    if not config.OCR_ENABLED:
        return ""
    try:
        import io
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=config.OCR_LANG) or ""
    except ImportError:
        log.warning("OCR skipped — install pytesseract + Pillow + Tesseract binary")
        return ""
    except Exception as exc:
        log.warning("OCR failed on page: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# PDF loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_pdf() -> list[dict]:
    """Return list of {text, page} dicts, one per non-empty page."""
    if not os.path.exists(config.PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {config.PDF_PATH}\n"
            "Drop knowledge.pdf into the data/ folder or set QNA_PDF_PATH."
        )
    doc = fitz.open(config.PDF_PATH)
    log.info("Loading PDF: %s  (%d pages)", config.PDF_PATH, len(doc))
    pages, ocr_count = [], 0
    for i, page in enumerate(doc, start=1):
        text = (page.get_text() or "").strip()
        if len(text) < _TEXT_THRESHOLD:
            ocr = _ocr_page(page).strip()
            if ocr:
                text, ocr_count = ocr, ocr_count + 1
        if text:
            pages.append({"text": text, "page": i})
    doc.close()
    log.info("Extracted %d pages  (%d via OCR)", len(pages), ocr_count)
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# RAG service
# ──────────────────────────────────────────────────────────────────────────────

class RAGService:
    """Cosine-similarity retrieval over a local FAISS index."""

    def __init__(self):
        self._index:  faiss.IndexFlatIP | None = None
        self._chunks: list[dict] = []         # [{text, page}, ...]
        self._model:  SentenceTransformer | None = None

    # ── embedding model ──────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (N, dim)."""
        if self._model is None:
            log.info("Loading embedding model '%s' on %s ...", config.EMBED_MODEL, config.EMBED_DEVICE)
            self._model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
            log.info("Embedding model ready")
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    # ── index build / load ───────────────────────────────────────────────────

    def build_or_load(self, force_rebuild: bool = False) -> None:
        if not force_rebuild and os.path.isfile(_INDEX_FILE) and os.path.isfile(_CHUNKS_FILE):
            try:
                self._index = faiss.read_index(_INDEX_FILE)
                with open(_CHUNKS_FILE, "rb") as f:
                    self._chunks = pickle.load(f)
                log.info("Loaded FAISS index (%d chunks) from %s", len(self._chunks), config.INDEX_DIR)
                return
            except Exception as exc:
                log.warning("Failed to load cached index (%s) — rebuilding", exc)

        # ── fresh build ──────────────────────────────────────────────────────
        pages = _load_pdf()
        if not pages:
            raise RuntimeError("PDF yielded no extractable text (even after OCR)")

        self._chunks = []
        for p in pages:
            for chunk_text in _split(p["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunk_text = chunk_text.strip()
                if chunk_text:
                    self._chunks.append({"text": chunk_text, "page": p["page"]})

        log.info(
            "Split into %d chunks (size=%d, overlap=%d)",
            len(self._chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP,
        )

        log.info("Embedding %d chunks with '%s' ...", len(self._chunks), config.EMBED_MODEL)
        vecs = self._embed([c["text"] for c in self._chunks])

        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalised vecs
        self._index.add(vecs)

        if os.path.isdir(config.INDEX_DIR):
            shutil.rmtree(config.INDEX_DIR)
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        faiss.write_index(self._index, _INDEX_FILE)
        with open(_CHUNKS_FILE, "wb") as f:
            pickle.dump(self._chunks, f)
        log.info("FAISS index saved to %s", config.INDEX_DIR)

    # ── retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[tuple[dict, float]]:
        """Return [(chunk, score), ...] sorted best-first.  score ∈ [0, 1]."""
        if self._index is None:
            self.build_or_load()
        q_vec = self._embed([query])                        # shape (1, dim)
        scores, indices = self._index.search(q_vec, config.TOP_K)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._chunks[idx], float(score)))
        return results

    def context_for(self, query: str) -> tuple[str, bool]:
        """Build a context string for the LLM.

        Returns (context_text, found).
        found=False means no chunk exceeded RAG_SCORE_THRESHOLD — caller should
        reply "not in document" instead of hallucinating.
        """
        hits = self.retrieve(query)
        if not hits or hits[0][1] < config.RAG_SCORE_THRESHOLD:
            log.info("RAG: no confident match (best=%.3f, threshold=%.3f)",
                     hits[0][1] if hits else 0.0, config.RAG_SCORE_THRESHOLD)
            return "", False
        parts = [f"[page {c['page']}]\n{c['text']}" for c, _ in hits]
        return "\n\n---\n\n".join(parts), True
