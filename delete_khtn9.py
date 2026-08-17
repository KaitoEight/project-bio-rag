"""Delete KHTN 9 chunks from VectorDB."""
from src.rag.vectorstore import VectorDB

vdb = VectorDB()
coll = vdb.db._collection
total = coll.count()
print(f"Total chunks before: {total}")

all_data = coll.get(include=["metadatas"])
ids = all_data.get("ids", [])
metas = all_data.get("metadatas", [])

khtn9_ids = []
for i, meta in enumerate(metas):
    if "KHTN 9" in meta.get("source", ""):
        khtn9_ids.append(ids[i])

print(f"KHTN 9 chunks found: {len(khtn9_ids)}")

if khtn9_ids:
    coll.delete(ids=khtn9_ids)
    print(f"Deleted {len(khtn9_ids)} chunks")

print(f"Remaining chunks: {coll.count()}")
