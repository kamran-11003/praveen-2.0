"""RAG service: PDF text extraction (with OCR fallback) + FAISS vector store + retrieval.

Flow at startup:
  1. Load PDF (PyMuPDF / fitz).
  2. For each page: extract embedded text. If empty/sparse → OCR the page image.
  3. Split combined text into overlapping chunks.
  4. Embed chunks with Gemini embeddings.
  5. Build / load a FAISS index (cached on disk).

Per query:
  retrieve(query) → list[str]   top-K chunks, concatenated context for the LLM.
"""

import os
import shutil

import fitz  # PyMuPDF

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


_TEXT_THRESHOLD = 40   # chars — below this we treat a page as "image only" and OCR it

_EMBED_BATCH = 80      # chunks per batch (free tier: 100 req/min)
_RATE_LIMIT_WAIT = 65  # seconds to wait when rate-limited


# ──────────────────────────────────────────────────────────────────────────────
# Batched embedding with rate-limit retry
# ──────────────────────────────────────────────────────────────────────────────

def _embed_with_retry(chunks: list, embeddings) -> FAISS:
    """Embed `chunks` in small batches, retrying on 429 rate-limit errors."""
    import time

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    all_vecs: list = []
    i = 0
    while i < len(texts):
        batch_texts = texts[i:i + _EMBED_BATCH]
        try:
            vecs = embeddings.embed_documents(batch_texts)
            all_vecs.extend(vecs)
            print(f"[rag]   Embedded {min(i + _EMBED_BATCH, len(texts))}/{len(texts)} chunks")
            i += _EMBED_BATCH
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                print(f"[rag]   Rate limited — waiting {_RATE_LIMIT_WAIT}s ...")
                time.sleep(_RATE_LIMIT_WAIT)
            else:
                raise

    # Build FAISS from pre-computed vectors
    text_embedding_pairs = list(zip(texts, all_vecs))
    store = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=metas)
    return store


# ──────────────────────────────────────────────────────────────────────────────
# OCR helper
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_page(page) -> str:
    """OCR a PyMuPDF page using Tesseract.  Returns "" if OCR is unavailable."""
    if not config.OCR_ENABLED:
        return ""
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page at 300 DPI for decent OCR accuracy
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=config.OCR_LANG) or ""
    except ImportError:
        print("[rag] OCR skipped: install pytesseract + Pillow and Tesseract binary")
        return ""
    except Exception as exc:
        print(f"[rag] OCR failed on page: {exc}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# PDF loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_pdf_text(pdf_path: str) -> list[Document]:
    """Extract text from every page; OCR pages that have no embedded text.

    Returns a list of LangChain Documents (one per page) with metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}.  "
            f"Drop your knowledge PDF there or set QNA_PDF_PATH."
        )

    docs: list[Document] = []
    doc = fitz.open(pdf_path)
    print(f"[rag] Loading PDF: {pdf_path}  ({len(doc)} pages)")

    ocr_pages = 0
    for i, page in enumerate(doc, start=1):
        text = (page.get_text() or "").strip()
        source = "embedded"
        if len(text) < _TEXT_THRESHOLD:
            ocr_text = _ocr_page(page).strip()
            if ocr_text:
                text = ocr_text
                source = "ocr"
                ocr_pages += 1
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={"page": i, "source": source},
        ))

    doc.close()
    print(f"[rag] Extracted {len(docs)} pages  ({ocr_pages} via OCR)")
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# RAG service
# ──────────────────────────────────────────────────────────────────────────────

class RAGService:
    """PDF-backed retrieval service with FAISS + Gemini embeddings."""

    def __init__(self):
        self._vectorstore: FAISS | None = None
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=config.GEMINI_EMBED_MODEL,
                google_api_key=config.GEMINI_API_KEY,
            )
        return self._embeddings

    def build_or_load(self, force_rebuild: bool = False) -> None:
        """Load cached FAISS index or build a fresh one from the PDF."""
        embeddings = self._get_embeddings()

        if not force_rebuild and os.path.isdir(config.INDEX_DIR):
            try:
                self._vectorstore = FAISS.load_local(
                    config.INDEX_DIR, embeddings, allow_dangerous_deserialization=True
                )
                print(f"[rag] Loaded cached FAISS index from {config.INDEX_DIR}")
                return
            except Exception as exc:
                print(f"[rag] Failed to load cached index ({exc}) — rebuilding")

        # Fresh build
        page_docs = _load_pdf_text(config.PDF_PATH)
        if not page_docs:
            raise RuntimeError("PDF produced no extractable text (even after OCR)")

        # Separators ordered from coarsest → finest. Urdu uses "۔" as sentence
        # terminator and "،" as comma — including these prevents mid-sentence
        # splits in Urdu / OCR'd pages.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "۔ ", "۔", ". ", "? ", "! ", "، ", ", ", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        chunks = splitter.split_documents(page_docs)
        print(f"[rag] Split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

        print("[rag] Embedding chunks with Gemini (batched to respect rate limits) ...")
        self._vectorstore = _embed_with_retry(chunks, embeddings)

        if os.path.isdir(config.INDEX_DIR):
            shutil.rmtree(config.INDEX_DIR)
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        self._vectorstore.save_local(config.INDEX_DIR)
        print(f"[rag] FAISS index saved to {config.INDEX_DIR}")

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Return top-K most relevant chunks for `query`."""
        if self._vectorstore is None:
            self.build_or_load()
        return self._vectorstore.similarity_search(query, k=k or config.TOP_K)

    def context_for(self, query: str) -> str:
        """Concatenate top-K chunks into a single context block for the LLM."""
        docs = self.retrieve(query)
        parts = []
        for d in docs:
            page = d.metadata.get("page", "?")
            parts.append(f"[page {page}]\n{d.page_content}")
        return "\n\n---\n\n".join(parts)
