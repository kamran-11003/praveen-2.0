Drop your knowledge PDF here as `knowledge.pdf`
(or set the QNA_PDF_PATH environment variable to a different location).

The first run will:
  1. Extract text from each page (PyMuPDF).
  2. OCR any image-only pages (requires Tesseract installed on PATH).
  3. Chunk the text and build a FAISS index here in ./faiss_index/

Delete the faiss_index/ folder to force a rebuild after replacing the PDF.
