"""
Book-level RAG: load full text from VectorDB (2113 chunks),
not from the filtered markdown export.
"""
import os
import asyncio
import unicodedata
import chromadb


def no_accent(text: str) -> str:
    """Remove Vietnamese diacritics for fuzzy matching."""
    n = unicodedata.normalize('NFD', text)
    return ''.join(c for c in n if unicodedata.category(c) != 'Mn')


def load_all_chunks() -> list[dict]:
    """Load all text chunks from vectorstore."""
    client = chromadb.PersistentClient(path="database")
    coll = client.get_collection("biology_text")
    result = coll.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        text = doc.strip()
        if len(text) < 100:
            continue
        chunks.append({
            "text": text,
            "source": meta.get("source", ""),
            "page": meta.get("page", "?"),
        })
    return chunks


def find_relevant_chunks(question: str, chunks: list, top_k: int = 10) -> list[dict]:
    """Find chunks matching question keywords (accent-insensitive)."""
    # Add biology-related keywords for biology questions
    bio_keywords = ["sinh", "te bao", "sinh hoc", "ho hap", "nuoi cay", "vat", "dong vat", "thuc vat",
                    "chuyen hoa", "nang luong", "trao doi", "sinh san", "sinh truong"]
    q_clean = no_accent(question.lower())
    extra = set(w for w in bio_keywords if w in q_clean)
    q_words = set(q_clean.split()) | extra
    relevant = []
    for ch in chunks:
        text_clean = no_accent(ch["text"].lower())
        score = sum(1 for w in q_words if len(w) > 2 and w in text_clean)
        if score > 0:
            relevant.append((score, ch))
    relevant.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in relevant[:top_k]]


async def answer_question(question: str) -> str:
    """Answer using Gemini with relevant chunks from vectorstore."""
    import importlib.util, os as _os
    spec = importlib.util.spec_from_file_location(
        "gemini_llm", _os.path.join(_os.path.dirname(__file__), "gemini_llm.py"))
    gemini_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemini_mod)
    GeminiLLMClient = gemini_mod.GeminiLLMClient

    chunks = load_all_chunks()
    relevant = find_relevant_chunks(question, chunks, top_k=15)

    if not relevant:
        return "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

    context_parts = []
    for ch in relevant:
        preview = ch["text"][:1500]
        context_parts.append(f"=== {ch['source']} trang {ch['page']} ===\n{preview}")

    context = "\n\n".join(context_parts)

    prompt = f"""Ban la tro ly AI mon Sinh hoc THCS. Tra loi CHI bang tieng Viet.

Dựa vào nội dung sách giáo khoa dưới đây, trả lời câu hỏi. Nếu câu hỏi không liên quan đến nội dung sách, trả lời: "Thông tin này không được đề cập trong sách giáo khoa."

---SÁCH GIÁO KHOA---
{context}

---CÂU HỎI---
{question}

QUY TẮC:
1. Chỉ dùng thông tin trong sách giáo khoa trên.
2. Trả lời ngắn gọn, đúng trọng tâm.
"""
# 3. Nếu không có trong sách: "Thông tin này không được đề cập trong sách giáo khoa."

    llm = GeminiLLMClient()
    resp = await llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=600,
    )
    return resp.strip()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks from vectorstore")
    q = "Ham ho hap te bao la gi? Cho biet cac chat tham gia va san pham cua qua trinh nay?"
    print(f"\nQ: {q}")
    ans = asyncio.run(answer_question(q))
    print(f"A: {ans}")
