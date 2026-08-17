"""Count and list chunks in ChromaDB per source PDF (Optimized)."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Set numpy monkeypatch
import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, str, bytes]
    }

from src.rag.vectorstore import VectorDB

vdb = VectorDB()
collection = vdb.db._collection

# Optimized count
total_count = collection.count()
print(f"Total chunks count via collection.count(): {total_count}")

# Fetch metadatas only to save memory and time
print("Fetching metadatas...")
results = collection.get(include=["metadatas"])
metadatas = results.get("metadatas", [])

print(f"Loaded {len(metadatas)} metadata records.")

# Group by source PDF
sources = {}
for meta in metadatas:
    if meta:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

print("\nChunks count per source file:")
for src, count in sorted(sources.items()):
    print(f"  Source: {src} | Chunks count: {count}")
