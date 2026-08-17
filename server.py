"""Standalone API server using book_rag + Gemini. No Flask needed."""
import asyncio
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

from dotenv import load_dotenv
load_dotenv()

# Check API key
if not os.environ.get("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Load book_rag once at startup
from src.rag.book_rag import load_all_chunks, find_relevant_chunks
from src.rag.gemini_llm import GeminiLLMClient

print("Loading chunks...")
CHUNKS = load_all_chunks()
print(f"Loaded {len(CHUNKS)} chunks")

print("Gemini client ready")
LLM = GeminiLLMClient()

LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)


async def get_answer(question: str) -> str:
    relevant = find_relevant_chunks(question, CHUNKS, top_k=15)
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
3. Nếu không có trong sách: "Thông tin này không được đề cập trong sách giáo khoa."
"""

    resp = await LLM.complete([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=600)
    return resp.strip()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                question = data.get("question", "")
            except Exception:
                question = ""

            if not question:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"question required"}')
                return

            print(f"Q: {question}")
            try:
                answer = LOOP.run_until_complete(get_answer(question))
                print(f"A: {answer[:100]}")
            except Exception as e:
                answer = f"Loi: {e}"
                print(f"Error: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"answer": answer, "images": []}, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # Suppress default logging


if __name__ == "__main__":
    port = 5000
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Server running on http://localhost:{port}")
    server.serve_forever()
