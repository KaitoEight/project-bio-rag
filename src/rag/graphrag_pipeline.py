import asyncio
import os
import json
import time
from pathlib import Path

PROMPT = (
    "Ban la chuyen gia Sinh hoc THCS. Trich xuat entities va relationships "
    "tu van ban sach giao khoa. Tra loi chi bang JSON: "
    '{"entities": [{"name": "...", "type": "...", "description": "..."}], '
    '"relationships": [{"source": "A", "target": "B", "type": "...", "description": "..."}]}. '
    "Neu khong co thi tra {\"entities\": [], \"relationships\": []}. "
    "---TAI LIEU---\n{text}"
)


class KG:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add(self, ents, rels, src):
        for e in ents:
            n = e.get("name", "").strip()
            if n and n not in self.nodes:
                self.nodes[n] = {"type": e.get("type", ""), "desc": e.get("description", ""), "sources": set()}
            if n:
                self.nodes[n]["sources"].add(src)
        for r in rels:
            s = r.get("source", "").strip()
            t = r.get("target", "").strip()
            if s and t and s != t:
                self.edges.append({"source": s, "target": t, "type": r.get("type", "RELATED_TO"), "desc": r.get("description", "")})

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"nodes": {k: {**v, "sources": list(v["sources"])} for k, v in self.nodes.items()}, "edges": self.edges}
        with open(path, "w", encoding="utf-8") as f2:
            json.dump(data, f2, ensure_ascii=False, indent=2)


async def call_with_retry(llm, messages, retries=5):
    """Call Gemini with exponential backoff on 429."""
    for attempt in range(retries):
        try:
            resp = await llm.complete(messages, temperature=0.1, max_tokens=2048)
            return resp
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = (2 ** attempt) * 5
                print(f"    429 rate limit, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


async def build():
    from src.rag.gemini_llm import GeminiLLMClient
    llm = GeminiLLMClient()
    kg = KG()
    files = list(Path("datanew/sgk_graphrag").glob("*.md"))
    print(f"Files: {len(files)}")
    success = 0
    errors = 0

    for f in files:
        text = f.read_text(encoding="utf-8")
        sections = []
        cur = ""
        for line in text.split("\n"):
            s = line.strip()
            if s == "---" or s.startswith("## Trang"):
                if cur.strip():
                    sections.append(cur.strip())
                cur = ""
            else:
                cur += " " + s
        if cur.strip():
            sections.append(cur.strip())
        print(f"  {f.name}: {len(sections)} sections")

        for i, sec in enumerate(sections):
            if len(sec) < 100:
                continue
            prompt = PROMPT.replace("{text}", sec[:2000])
            try:
                resp = await call_with_retry(llm, [{"role": "user", "content": prompt}])
                txt = resp.strip()
                # strip markdown code fences
                while txt.startswith("`"):
                    txt = txt[1:]
                idx = txt.find("\n")
                if idx >= 0:
                    first = txt[:idx].strip()
                    if first in ("json", "JSON", "{"):
                        txt = txt[idx+1:].strip()
                while txt.startswith("`"):
                    txt = txt[1:]
                txt = txt.strip("`").strip()
                j = json.loads(txt)
                kg.add(j.get("entities", []), j.get("relationships", []), f.name)
                success += 1
                if success % 50 == 0:
                    print(f"    ... {success} sections done")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    err {i}: {e}")

    print(f"\nDone! Success: {success}, Errors: {errors}")
    print(f"Nodes: {len(kg.nodes)}, Edges: {len(kg.edges)}")
    kg.save("database/graphrag/kg.json")
    print("Saved!")
    return kg


if __name__ == "__main__":
    asyncio.run(build())
