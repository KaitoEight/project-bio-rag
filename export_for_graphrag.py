"""Export text chunks from VectorDB to markdown files for GraphRAG ingestion."""
import os
import asyncio
from pathlib import Path
from src.rag.vectorstore import VectorDB

OUT_DIR = Path("datanew/sgk_graphrag")
OUT_DIR.mkdir(exist_ok=True)

vdb = VectorDB()
coll = vdb.db._collection
all_data = coll.get(include=["documents", "metadatas"])

ids = all_data.get("ids", [])
docs = all_data.get("documents", [])
metas = all_data.get("metadatas", [])

# Group by source file
files = {}
for i, doc in enumerate(docs):
    meta = metas[i]
    source = meta.get("source", "unknown")
    if source not in files:
        files[source] = []
    files[source].append((meta.get("page", "?"), doc))

print(f"Exporting {len(docs)} chunks from {len(files)} files...")

for source, chunks in files.items():
    safe_name = source.replace("/", "_").replace("\\", "_").replace(":", "")
    chunks.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
    out_path = OUT_DIR / f"{safe_name}.md"

    lines = [f"# {source}\n"]
    for page, text in chunks:
        if len(text.strip()) < 50:
            continue
        lines.append(f"\n## Trang {page}\n")
        lines.append(text.strip())
        lines.append("\n---")

    content = "\n".join(lines)
    out_path.write_text(content, encoding="utf-8")
    print(f"  Wrote: {out_path.name} ({len(content)} chars, {len(chunks)} chunks)")

print(f"\nDone! Files in: {OUT_DIR}")
