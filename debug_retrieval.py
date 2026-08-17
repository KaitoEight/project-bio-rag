"""Debug script to inspect retrieved contexts and raw LLM answers."""
import sys
import os
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
load_dotenv()

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
from src.rag.llm import get_llm

vdb = VectorDB()
retriever = vdb.get_retriever()
llm = get_llm()

questions = [
    "Năng lượng ánh sáng đã chuyển hoá thành dạng năng lượng nào?",
    "Nguyên tử có cấu tạo như thế nào và gồm các hạt nào?",
    "Định luật bảo toàn khối lượng được phát biểu như thế nào?",
    "Khái niệm về gene và cấu trúc của DNA là gì?"
]

for idx, q in enumerate(questions):
    print(f"\n============================================================")
    print(f"[CÂU HỎI {idx+1}]: {q}")
    print(f"============================================================")
    
    docs = retriever.invoke(q)
    print(f"-> Đã tìm thấy {len(docs)} đoạn tài liệu liên quan:\n")
    
    for d_idx, doc in enumerate(docs):
        print(f"--- [Đoạn {d_idx+1}] Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')} ---")
        print(doc.page_content[:400])
        print("-" * 40)
        
    # Let's see the raw output of the LLM without parser!
    # Build prompt manually
    context_text = "\n\n".join([doc.page_content.strip() for doc in docs])
    prompt = f"""<|im_start|>system
Bạn là trợ lý AI môn Sinh học THCS. Bạn PHẢI trả lời hoàn toàn bằng TIẾNG VIỆT.<|im_end|>
<|im_start|>user
[TÀI LIỆU SÁCH GIÁO KHOA]:
{context_text}

[CÂU HỎI]:
{q}

[QUY TẮC NGHIÊM NGẶT]:
1. CHỈ dùng thông tin trong tài liệu trên. KHÔNG tự suy diễn, KHÔNG bịa.
2. CHỈ trả lời ĐÚNG nội dung được hỏi. KHÔNG thêm thông tin không liên quan đến câu hỏi.
3. KHÔNG ghép nối các thông tin rời rạc từ những đoạn khác chủ đề để tạo câu trả lời.
4. Nếu một đoạn tài liệu không liên quan đến câu hỏi, hãy BỎ QUA đoạn đó.
5. Nếu tài liệu không chứa câu trả lời, hãy trả lời ĐÚNG CÂU SAU: "Thông tin này không được đề cập trong sách giáo khoa."<|im_end|>
<|im_start|>assistant
"""
    print("\n--- RAW LLM RESPONSE ---")
    try:
        resp = llm.invoke(prompt)
        # Handle message vs string return depending on langchain version
        text_resp = resp.content if hasattr(resp, 'content') else str(resp)
        print(text_resp)
    except Exception as e:
        print("LLM Error:", e)
