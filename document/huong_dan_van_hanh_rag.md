# Hướng Dẫn Vận Hành Project RAG

Tài liệu này mô tả:
- Các bước chạy hệ thống từ đầu đến cuối.
- Ý nghĩa từng lệnh CLI.
- Quy trình review ảnh thủ công và các thao tác CRUD metadata mới.

---

## 1) Mục tiêu hệ thống

Project xử lý dữ liệu SGK thành 2 lớp:
1. `Text index`: phục vụ trả lời nội dung văn bản.
2. `Image index`: phục vụ tìm ảnh minh họa liên quan câu hỏi.

Ngoài ra có cơ chế:
- Extract ảnh từ page.
- Export ảnh ra file review.
- Người dùng sửa caption/xóa ảnh không phù hợp.
- Upsert một item mới thủ công (manual add).
- Apply lại vào vector DB.

---

## 2) Chuẩn bị môi trường

### 2.1 Cài dependencies

```bash
pip install -r requirements.txt
```

### 2.1b Nếu chạy trên Google Colab

Cần cài thêm thư viện hệ thống cho `pdf2image` và OCR tiếng Việt:

```python
!apt-get update
!apt-get install -y poppler-utils
!apt-get install -y tesseract-ocr tesseract-ocr-vie
```

Để tránh mất DB/checkpoint khi runtime Colab bị ngắt, mount Google Drive và trỏ `RAG_DATABASE_DIR` vào Drive trước khi chạy ETL:

```python
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["RAG_DATABASE_DIR"] = "/content/drive/MyDrive/project_bio_rag/database"
```

Khi đó các file sau sẽ nằm trong Drive thay vì runtime `/content`:
- `chroma.sqlite3` và các collection Chroma.
- `processed_files.txt`.
- `images/...`.
- `image_review_manifest.jsonl`.
- `image_caption_cache.json`.

Nếu PDF đầu vào cũng đặt trên Drive, trỏ thêm:

```python
os.environ["RAG_DATA_DIR"] = "/content/drive/MyDrive/project_bio_rag/data"
```

Không xóa `%rm -rf database` trong runtime nếu mục tiêu là resume; lệnh này chỉ xóa thư mục local và không reset DB đang nằm trên Drive.

### 2.2 Tạo file `.env`

```bash
cp .env.example .env
```

### 2.3 Biến quan trọng

- `HF_TOKEN`: bắt buộc để tải model từ Hugging Face.
- `USE_GPU=true|false`: bật/tắt GPU.
- `TESSERACT_CMD`: đường dẫn `tesseract.exe` trên Windows, ví dụ `C:/Program Files/Tesseract-OCR/tesseract.exe`.
- `POPPLER_PATH`: đường dẫn thư mục `bin` của Poppler trên Windows, ví dụ `C:/poppler/Library/bin`.
- `RAG_DATABASE_DIR`: nơi lưu thư mục database. Để rỗng thì dùng `./database`; trên Colab nên trỏ vào Google Drive.
- `RAG_DATA_DIR`: nơi đọc PDF đầu vào. Để rỗng thì dùng `./data`.
- `IMAGE_CAPTION_ENABLED=true|false`: bật/tắt caption model cho ảnh.
- `IMAGE_EXTRACTION_VERSION=v4_owlvit_context_caption`: version thuật toán extract/caption ảnh, đổi version sẽ buộc reprocess image page.
- `IMAGE_REVIEW_MANIFEST_PATH`: nơi lưu manifest review ảnh. Để rỗng thì mặc định nằm trong `RAG_DATABASE_DIR`.

Nếu chưa biết máy cài Poppler/Tesseract ở đâu, repo có sẵn:

```text
windows_tools/poppler.zip
windows_tools/tesseract-ocr.zip
```

Giải nén và cấu hình theo `document/windows_tools_setup.md`.

Link tải nếu thiếu zip:
- Poppler Windows releases: https://github.com/oschwartz10612/poppler-windows/releases/
- Hướng dẫn Poppler cho `pdf2image`: https://pdf2image.readthedocs.io/en/latest/installation.html
- Tesseract Windows UB Mannheim: https://github.com/UB-Mannheim/tesseract/wiki
- Tài liệu Tesseract Windows UB Mannheim: https://ub-mannheim.github.io/Tesseract_Dokumentation/Tesseract_Doku_Windows.html

---

## 3) Danh sách lệnh và mục đích

## `python main.py --text-only`
- Chạy ETL cho text.

## `python main.py --image-only`
- Chạy ETL cho image.

## `python main.py --etl`
- Chạy full pipeline text + image trong 1 lệnh.

## `python main.py --export-image-review <output.json>`
- Export danh sách ảnh cho human review.
- Mặc định chỉ export item `pending`.

## `python main.py --export-image-review <output.json> --review-pdf "<file.pdf>"`
- Export review theo 1 PDF cụ thể.

## `python main.py --export-image-review <output.json> --review-include-completed`
- Export cả item đã review (`approved/rejected`).

## `python main.py --export-image-db <output.json>`
- Export snapshot metadata DB hiện có (đọc từ manifest).

## `python main.py --export-image-db <output.json> --review-pdf "<file.pdf>"`
- Export snapshot metadata DB theo 1 PDF.

## `python main.py --upsert-image-review-item <item.json> --review-user <name>`
- Upsert 1 item metadata theo `image_id`:
  - `image_id` đã có -> update
  - `image_id` chưa có -> create mới
- Sau upsert sẽ sync sang image vector DB.

## `python main.py --apply-image-review <review.json> --review-user <name>`
- Apply hàng loạt từ file JSON array vào DB.
- Mặc định cho phép tạo mới item nếu `image_id` chưa tồn tại.

## `python main.py --apply-image-review <review.json> --review-pdf "<file.pdf>" --review-user <name>`
- Apply hàng loạt nhưng chỉ cho item thuộc `pdf_filename` tương ứng.

## `python main.py --replace-image-db <snapshot.json> --review-user <name>`
- Replace toàn bộ manifest ảnh theo JSON array snapshot.
- Rebuild image vector index theo snapshot mới.
- Item không còn trong snapshot sẽ bị xóa khỏi manifest và image index.

## `python main.py --api --port 5000`
- Mở Flask API server để frontend gọi chat, ETL và image review.

---

## 4) Quy tắc review và upsert (quan trọng)

1. `--apply-image-review` là **upsert theo item có trong file JSON**, không phải sync full theo file.
2. Nếu bạn **xóa item khỏi array review JSON**, item đó **không tự bị xóa** khỏi DB.
3. Muốn loại ảnh khỏi retrieval, đặt một trong các cách:
   - `review_status = "rejected"` hoặc `"deleted"`
   - `is_active = false`
   - `delete = true`
4. `caption_vi_manual` và `keywords_vi_manual` là nguồn ưu tiên để tạo `final_caption_vi`, `final_keywords_vi`.
5. `--replace-image-db` là sync full theo file snapshot: xóa item khỏi array JSON đồng nghĩa xóa item khỏi manifest và image index.
6. Nếu thêm ảnh mới thủ công, nên bảo đảm `image_path` trỏ đúng file ảnh tồn tại.

---

## 5) Quy trình khuyến nghị

## Kịch bản A: Khởi tạo mới hoàn toàn

1. Xóa DB cũ:

```powershell
Remove-Item -Recurse -Force "D:\personal_repo\project_rag\database"
New-Item -ItemType Directory -Path "D:\personal_repo\project_rag\database" | Out-Null
```

2. ETL text:

```bash
python main.py --text-only
```

3. ETL image:

```bash
python main.py --image-only
```

4. Export file review:

```bash
python main.py --export-image-review database/review_images.json
```

5. Người dùng cập nhật thủ công `database/review_images.json`.

6. Apply review:

```bash
python main.py --apply-image-review database/review_images.json --review-user charlie
```

7. Chạy API:

```bash
python main.py --api --port 5000
```

## Kịch bản B: Build nhanh 1 lệnh

```bash
python main.py --etl
```

## Kịch bản C: Chỉ review một tài liệu

```bash
python main.py --export-image-review database/review_sgk6.json --review-pdf "SGK KHTN 6 CD.pdf"
python main.py --apply-image-review database/review_sgk6.json --review-pdf "SGK KHTN 6 CD.pdf" --review-user charlie
```

## Kịch bản D: Upsert một item mới/cập nhật một item

```bash
python main.py --upsert-image-review-item database/one_item.json --review-user charlie
```

Ví dụ `database/one_item.json`:

```json
{
  "image_id": "manual_0001",
  "pdf_filename": "SGK KHTN 6 CD.pdf",
  "page_number": 88,
  "image_path": "D:/personal_repo/project_rag/database/images/SGK KHTN 6 CD/page_88_img_manual_1.png",
  "page_snapshot_path": "D:/personal_repo/project_rag/database/images/SGK KHTN 6 CD/pages/page_88_snapshot.png",
  "bbox": "0,0,100,100",
  "figure_label": "Hình bổ sung",
  "figure_caption": "Mô tả bổ sung thủ công",
  "caption_vi_manual": "Hai con hải mã trên tảng băng",
  "keywords_vi_manual": "hải mã, vùng cực, băng tuyết",
  "review_status": "edited",
  "is_active": true,
  "review_notes": "added manually"
}
```

## Kịch bản E: Export DB, chỉnh JSON, rồi replace toàn bộ DB theo file

```bash
python main.py --export-image-db database/all_image_db.json
python main.py --replace-image-db database/all_image_db.json --review-user charlie
```

Payload `database/all_image_db.json` là JSON array. Ví dụ rút gọn:

```json
[
  {
    "image_id": "manual_0001",
    "pdf_filename": "SGK KHTN 6 CD.pdf",
    "page_number": 88,
    "image_path": "D:/personal_repo/project_rag/database/images/SGK KHTN 6 CD/page_88_img_manual_1.png",
    "page_snapshot_path": "D:/personal_repo/project_rag/database/images/SGK KHTN 6 CD/pages/page_88_snapshot.png",
    "bbox": "0,0,100,100",
    "figure_label": "Hình bổ sung",
    "figure_caption": "Mô tả bổ sung thủ công",
    "caption_vi": "Mô tả tự động nếu có",
    "keywords_vi": "từ khóa tự động",
    "caption_vi_manual": "Hai con hải mã trên tảng băng",
    "keywords_vi_manual": "hải mã, vùng cực, băng tuyết",
    "review_status": "edited",
    "is_active": true,
    "review_notes": "kept by snapshot"
  },
  {
    "image_id": "bad_image_0001",
    "review_status": "rejected",
    "is_active": false,
    "review_notes": "giữ metadata nhưng loại khỏi retrieval"
  }
]
```

Lưu ý:
- Nếu object bị xóa khỏi array rồi chạy `--replace-image-db`, object đó bị xóa khỏi manifest.
- Nếu object còn trong array nhưng `review_status=rejected|deleted` hoặc `is_active=false`, object vẫn còn trong manifest nhưng bị xóa khỏi image index.

---

## 6) Reset và xử lý sự cố

Script `reset_status.py`:

## Reset trạng thái image cho tất cả page
```bash
python reset_status.py --all
```

## Xóa image vector collections
```bash
python reset_status.py --image-index
```

## Reset cả status + image index
```bash
python reset_status.py --images-full
```

## Reset image status theo 1 file PDF
```bash
python reset_status.py "D:/personal_repo/project_rag/data/SGK KHTN 6 CD.pdf"
```

---

## 7) Kiểm tra sau khi chạy

Xác nhận các file/thư mục:
- `database/chroma.sqlite3`
- `database/images/...`
- `database/image_review_manifest.jsonl`

Kiểm tra nhanh CLI:

```bash
python main.py --help
```

---

## 8) Gợi ý vận hành ổn định

- Nếu muốn tăng tốc ETL ảnh: tạm để `IMAGE_CAPTION_ENABLED=false` và dùng flow review thủ công.
- Khi đổi thuật toán extract ảnh, tăng `IMAGE_EXTRACTION_VERSION` để hệ thống tự biết cần reprocess page image.
- Nên chạy theo thứ tự: `text-only` -> `image-only` -> `export/apply review` -> `app`.
- Với thao tác CRUD metadata, ưu tiên `--upsert-image-review-item` và `--apply-image-review` thay vì sửa trực tiếp DB SQLite.
