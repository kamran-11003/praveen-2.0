"""Quick RAG smoke-test — one question per institution in the PAK Center catalogue."""

import sys
import config
from rag_service import RAGService

QUESTIONS = [
    # Institution          Question
    ("NADRA",              "CNIC renewal ke liye kya documents chahiye?"),
    ("DGI&P (Passports)",  "Passport renewal ke liye kya documents chahiye?"),
    ("Islamabad Police",   "Character certificate kaise milta hai aur kitna waqt lagta hai?"),
    ("IESCO",              "New electricity connection ke liye apply kaise karein?"),
    ("CDA",                "Building plan approval ke liye kya chahiye?"),
    ("MOFA",               "Apostille attestation kya hoti hai aur kya documents chahiye?"),
    ("DC Office",          "Domicile certificate ke liye kya documents aur fees hain?"),
]

def main():
    print("=" * 70)
    print("  PAK Center QnA — RAG Smoke Test")
    print("  Model   :", config.GEMINI_MODEL)
    print("  Embed   :", config.GEMINI_EMBED_MODEL)
    print("=" * 70)

    print("\n[1/2] Building / loading FAISS index …")
    rag = RAGService()
    rag.build_or_load()
    print("[1/2] Index ready.\n")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.3,
    )

    SYSTEM = (
        "You are a friendly assistant for PAK Center Islamabad. "
        "Answer ONLY from the CONTEXT provided. Be concise and conversational — "
        "the answer will be spoken aloud. Use the same language as the question."
    )

    print("[2/2] Running queries …\n")
    for inst, question in QUESTIONS:
        print("─" * 70)
        print(f"  Institution : {inst}")
        print(f"  Question    : {question}")

        try:
            context = rag.context_for(question)
            user_block = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
            result = llm.invoke([
                SystemMessage(content=SYSTEM),
                HumanMessage(content=user_block),
            ])
            answer = result.content.strip()
            print(f"  Answer      : {answer}")
        except Exception as exc:
            print(f"  ERROR       : {exc}", file=sys.stderr)

        print()

    print("=" * 70)
    print("  Test complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
