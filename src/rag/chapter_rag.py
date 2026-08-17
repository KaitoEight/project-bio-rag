"""
Chapter-based RAG: index whole chapters (Bài) instead of chunks.
- Each chapter = full text + vector embedding
- On query: find top-k relevant chapters → pass full chapter text to Gemini
- Pros: full context, no chunking loss, works even with OCR errors
"""
import os
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


CHROMA_DIR = "database/chromadb_chapters"


def parse_chapters(text: str, source: str) -> list[dict]:
    """Split text into chapters by 'Bài N' heading pattern."""
    chapters = []
    # Split by "Bài N" headings (with page markers removed)
    pattern = re.compile(r"(Bài \d+[^\n—–-]{0,150})", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        content = re.sub(r"## Trang \d+\n?", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = content.strip()

        if len(content) < 200:
            continue

        chapter_id = f"{source}||{title}"
        chapters.append({
            "id": chapter_id,
            "title": title,
            "content": content,
            "source": source,
            "char_count": len(content),
        })
    return chapters


def build_chapter_index():
    """Build chapter index from markdown files."""
    print("Building chapter index...")
    all_chapters = []

    for f in Path("datanew/sgk_graphrag").glob("*.md"):
        text = f.read_text(encoding="utf-8")
        chapters = parse_chapters(text, f.name)
        print(f"  {f.name}: {len(chapters)} chapters")
        all_chapters.extend(chapters)

    print(f"Total chapters: {len(all_chapters)}")
    if not all_chapters:
        print("No chapters found!")
        return

    # Embed chapters
    print("Embedding chapters...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = [c["title"] + "\n" + c["content"][:3000] for c in all_chapters]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Store in Chroma
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("chapters")
    except Exception:
        pass
    coll = client.create_collection("chapters")

    ids = [c["id"] for c in all_chapters]
    metas = [{"title": c["title"], "source": c["source"], "char_count": c["char_count"]} for c in all_chapters]
    docs = [c["content"] for c in all_chapters]

    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings.tolist())
    print(f"Saved {len(all_chapters)} chapters to {CHROMA_DIR}")


def load_index():
    """Load chapter index."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_collection("chapters")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return coll, model


async def query_chapters(question: str, top_k: int = 3) -> list[dict]:
    """Find top-k relevant chapters for a question."""
    coll, model = load_index()
    q_emb = model.encode([question])
    results = coll.query(query_embeddings=q_emb.tolist(), n_results=top_k)
    chapters = []
    for i in range(len(results["documents"][0])):
        chapters.append({
            "title": results["metadatas"][0][i]["title"],
            "source": results["metadatas"][0][i]["source"],
            "content": results["documents"][0][i],
        })
    return chapters


if __name__ == "__main__":
    build_chapter_index()
