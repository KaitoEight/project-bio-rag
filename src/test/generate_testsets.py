"""Sinh bộ test đánh giá RAG — MỖI SÁCH MỘT BỘ (12 PDF -> 12 file CSV).

Ý tưởng cốt lõi để đo được Precision/Recall/Rank một cách XÁC ĐỊNH:
    - Với mỗi sách, chọn ngẫu nhiên (có seed, lặp lại được) một số TRANG nội dung.
    - OCR trang đó, đưa text cho Gemini 2.5 Flash để đặt 1 câu hỏi + đáp án chuẩn
      (ground truth) CHỈ dựa trên trang đó.
    - Lưu kèm (source_book, source_page) -> đây là "tài liệu vàng" để sau này
      đối chiếu metadata chunk mà hệ RAG truy xuất được.

Khác với cách Ragas cũ: ta chủ động biết câu hỏi đến từ trang nào, nên tính được
Precision@k / Recall@k / MRR thật, không phụ thuộc LLM chấm.

Yêu cầu: GOOGLE_API_KEY trong .env (đã có). Tesseract cho OCR tiếng Việt.

Chạy:
    python src/test/generate_testsets.py
"""

import os
import re
import sys
import csv
import glob
import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor

# Cho phép import package `src` khi chạy trực tiếp.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:  # đảm bảo in được tiếng Việt trên console Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pytesseract
from dotenv import load_dotenv

from src.config import TESSERACT_CMD, DATA_DIR, PERSIST_DIR
from src.etl.cleaner import clean_vietnamese_text
from src.test.eval_llm import get_eval_llm, is_configured, config_help

load_dotenv()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
logging.basicConfig(level=logging.WARNING)

# ----- Tham số sinh test (chỉnh ở đây) ---------------------------------------
NUM_QUESTIONS_PER_BOOK = int(os.getenv("EVAL_QUESTIONS_PER_BOOK", "12"))
SKIP_FRONT_PAGES = 10          # bỏ bìa / mục lục / lời nói đầu
MIN_PAGE_TEXT_LEN = 600        # trang phải đủ chữ mới đặt câu hỏi
MAX_PAGE_CANDIDATES = 60       # số trang tối đa xét cho mỗi sách (đủ dư)
RANDOM_SEED = 42               # cố định để bộ test lặp lại được

GEN_PROMPT = """Bạn là giáo viên ra đề môn Khoa học tự nhiên THCS.
Dưới đây là nội dung OCR của MỘT trang sách giáo khoa.

[NỘI DUNG TRANG]:
{page_text}

Hãy tạo MỘT câu hỏi kiểm tra kiến thức mà học sinh có thể trả lời ĐƯỢC HOÀN TOÀN
chỉ dựa vào nội dung trang trên, kèm đáp án chuẩn (ground truth) ngắn gọn, chính xác.

Quy tắc:
- Câu hỏi phải cụ thể, rõ ràng, bằng tiếng Việt, KHÔNG tham chiếu "trang này"/"đoạn trên".
- Đáp án chỉ lấy từ nội dung trang, không bịa thêm.
- Nếu trang chủ yếu là bìa, mục lục, hình ảnh trang trí, hoặc quá ít thông tin để
  ra một câu hỏi kiến thức tốt -> đặt "answerable": false.

CHỈ trả về JSON thuần (không markdown), đúng định dạng:
{{"answerable": true, "question": "...", "ground_truth": "..."}}
"""


def _page_num(path: str) -> int:
    m = re.search(r"page_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _ocr_page(image_path: str) -> str:
    raw = pytesseract.image_to_string(image_path, lang="vie")
    return clean_vietnamese_text(raw)


def _parse_json(text: str) -> dict:
    """Bóc JSON từ output Gemini (có thể bọc ```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def generate_for_book(pdf_path: str, output_dir: str, llm) -> str:
    base_name = os.path.basename(pdf_path)
    book_name = base_name[:-4] if base_name.lower().endswith(".pdf") else base_name
    csv_path = os.path.join(output_dir, f"{book_name}_testset.csv")

    if os.path.exists(csv_path):
        print(f"  -> Đã có {os.path.basename(csv_path)}, bỏ qua.")
        return csv_path

    pages_dir = os.path.join(str(PERSIST_DIR), "images", book_name, "pages")
    images = sorted(glob.glob(os.path.join(pages_dir, "*.png")), key=_page_num)
    images = [p for p in images if _page_num(p) > SKIP_FRONT_PAGES]
    if not images:
        print(f"  !! Không tìm thấy ảnh trang cho '{book_name}' tại {pages_dir}")
        return ""

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)
    candidates = images[:MAX_PAGE_CANDIDATES]

    # OCR song song các trang ứng viên.
    with ThreadPoolExecutor(max_workers=8) as ex:
        ocr_texts = list(ex.map(_ocr_page, candidates))

    rows = []
    for img_path, page_text in zip(candidates, ocr_texts):
        if len(rows) >= NUM_QUESTIONS_PER_BOOK:
            break
        if not page_text or len(page_text) < MIN_PAGE_TEXT_LEN:
            continue
        page_no = _page_num(img_path)
        try:
            resp = llm.invoke(GEN_PROMPT.format(page_text=page_text[:4000]))
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _parse_json(content)
        except Exception as exc:
            logging.warning("Sinh câu hỏi lỗi (trang %s): %s", page_no, exc)
            continue

        if not data.get("answerable") or not data.get("question") or not data.get("ground_truth"):
            continue
        rows.append({
            "question": data["question"].strip(),
            "ground_truth": data["ground_truth"].strip(),
            "source_book": base_name,
            "source_page": page_no,
        })
        print(f"  [{len(rows):>2}/{NUM_QUESTIONS_PER_BOOK}] trang {page_no}: {data['question'][:70]}")

    if not rows:
        print(f"  !! Không sinh được câu hỏi nào cho {book_name}.")
        return ""

    # utf-8-sig (có BOM) để Excel hiển thị đúng tiếng Việt, không ra ô vuông.
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "source_book", "source_page"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Đã lưu {len(rows)} câu: {os.path.basename(csv_path)}")
    return csv_path


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "testsets")
    os.makedirs(output_dir, exist_ok=True)

    if not is_configured():
        print(config_help())
        return

    llm = get_eval_llm(temperature=0.7)

    pdf_files = sorted(glob.glob(os.path.join(str(DATA_DIR), "*.pdf")))
    print(f"Tìm thấy {len(pdf_files)} sách. Sinh {NUM_QUESTIONS_PER_BOOK} câu/sách.\n")

    for pdf_path in pdf_files:
        print(f"== {os.path.basename(pdf_path)} ==")
        try:
            generate_for_book(pdf_path, output_dir, llm)
        except Exception as exc:
            import traceback
            print(f"  !! Lỗi: {exc}")
            traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
