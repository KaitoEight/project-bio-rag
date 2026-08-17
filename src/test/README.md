# Bộ đánh giá hệ thống RAG (12 sách)

Đánh giá hệ Trợ lý ảo Khoa học tự nhiên: **chất lượng truy xuất (RAG)** bằng số liệu
IR xác định (Precision / Recall / MRR) + **chất lượng câu trả lời của Qwen 2.5** được
**một LLM thứ 2 chấm lại**.

Mỗi cuốn trong 12 PDF có **một bộ test riêng** → 12 file CSV → xếp hạng 12 sách.

## 1. Yêu cầu

```bash
pip install -r requirements.txt   # đã gồm langchain-openai, pandas, pytesseract
```

- Tesseract tiếng Việt (`TESSERACT_CMD` trong `.env`) cho bước sinh test.
- Vector DB đã build sẵn trong `database/` (collection `biology_text`, ~13.7k chunk).
- **LLM đánh giá** (sinh test + chấm) cấu hình qua `.env`, dùng endpoint
  **OpenAI-compatible** nên cắm được MiMo / Groq / OpenRouter / vLLM tự host:

  ```env
  EVAL_LLM_BASE_URL=https://api.<...>/v1   # Groq: https://api.groq.com/openai/v1
  EVAL_LLM_API_KEY=<token của bạn>
  EVAL_LLM_MODEL=<model id>                 # vd mimo-2.5-pro / llama-3.3-70b-versatile
  ```
  (Logic xây client ở `eval_llm.py`. Nếu key nằm ở biến tên khác, đặt
  `EVAL_LLM_API_KEY_ENV=GROQ_API_KEY` chẳng hạn.)

## 2. Vì sao đo được Precision / Recall / Rank "thật"?

Mỗi câu hỏi được sinh **từ một trang cụ thể** của một sách, nên ta biết chắc
"tài liệu vàng" = `(source_book, source_page)`. Khi chạy hệ RAG, ta đối chiếu
**metadata** (`source`, `page`) của các chunk truy xuất được với tài liệu vàng:

| Số liệu | Ý nghĩa |
|---|---|
| **Precision@k (page)** | Tỷ lệ chunk truy xuất đúng trang nguồn. |
| **Recall@k (page)** = hit@k | Có tìm được trang nguồn trong top-k không (0/1, lấy trung bình). |
| **MRR (page)** | Điểm rank = 1/(thứ hạng chunk đúng đầu tiên). |
| **\*_book** | Bản book-level (chỉ cần đúng sách) → đo nhiễu chéo giữa 12 sách. |
| `recall_top10_diag` | "Trần recall" top-10 thô (bỏ qua relevance gate) để chẩn đoán. |

Số liệu này **không phụ thuộc LLM chấm** → minh bạch, lặp lại được. (Cho phép sai số
±1 trang vì chunk có thể tràn qua ranh giới trang — xem `metrics.py`.)

## 3. Các bước

### Bước 1 — Sinh 12 bộ test
`generate_testsets.py` chọn ngẫu nhiên (seed cố định) các trang nội dung của từng sách,
OCR, rồi dùng Gemini đặt câu hỏi + đáp án chuẩn, lưu kèm `(source_book, source_page)`.

```bash
python src/test/generate_testsets.py
```
→ Sinh `src/test/testsets/<tên sách>_testset.csv` (mặc định 12 câu/sách,
chỉnh bằng env `EVAL_QUESTIONS_PER_BOOK`).

### Bước 2 — Chạy đánh giá + xếp hạng
`evaluator.py` đưa từng câu hỏi vào **đúng pipeline Qwen 2.5 thật**
(`AppServices` → `HybridRetriever` → prompt → Qwen 2.5 → parser), tính số liệu IR,
rồi để **Gemini chấm lại** câu trả lời (correctness / faithfulness / relevancy, 1–5).

```bash
python src/test/evaluator.py
```
Kết quả:
- `testsets/<sách>_result.csv` — chi tiết từng câu.
- `evaluation_report.csv` — tổng hợp + **xếp hạng** 12 sách.
- `evaluation_report.md` — bảng leaderboard dễ đọc.

> Lần chạy đầu hơi chậm vì nạp model Qwen 2.5 + vector DB (cache lại cho các câu sau).

## 4. Tinh chỉnh (biến môi trường tùy chọn)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `EVAL_QUESTIONS_PER_BOOK` | 12 | Số câu hỏi mỗi sách. |
| `EVAL_LLM_BASE_URL` | — | Endpoint OpenAI-compatible (MiMo/Groq/...). |
| `EVAL_LLM_API_KEY` | — | Token của bạn. |
| `EVAL_LLM_MODEL` | — | Model id dùng để sinh test + chấm. |

## 5. File trong thư mục

- `metrics.py` — hàm tính Precision@k / Recall@k / MRR (page & book level).
- `generate_testsets.py` — sinh 12 bộ test có ground-truth nguồn.
- `evaluator.py` — chạy RAG thật, đo IR, Gemini chấm lại, xếp hạng.
