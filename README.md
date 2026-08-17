# Biology RAG (Text + Image)

Dự án RAG cho SGK KHTN/Sinh học với 2 pipeline chính:
- ETL text: OCR tiếng Việt -> chunk -> index vào Chroma.
- ETL image: detect/crop ảnh theo từng page -> enrich metadata -> index image + metadata.

Hệ thống hỗ trợ human-review và CRUD metadata ảnh để tăng chất lượng image retrieval.

## 1) Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
```

Thiết lập tối thiểu trong `.env`:
- `HF_TOKEN=<your_token>`
- `USE_GPU=true` (nếu có GPU)

Nếu chạy OCR trên Windows, khai báo thêm:

```env
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
POPPLER_PATH=C:/poppler/Library/bin
```

Repo có sẵn zip local trong `windows_tools/`:

```text
windows_tools/poppler.zip
windows_tools/tesseract-ocr.zip
```

Nếu chưa biết máy cài Poppler/Tesseract ở đâu, xem hướng dẫn giải nén và khai báo path tại `document/windows_tools_setup.md`.

Nếu chạy trên Google Colab, cài thêm system libs trước khi ETL:

```python
!apt-get update
!apt-get install -y poppler-utils
!apt-get install -y tesseract-ocr tesseract-ocr-vie
```

### Lưu `database` trên Google Drive khi chạy Colab

Mặc định project ghi Chroma DB, checkpoint ETL, ảnh crop, manifest review và caption cache vào `./database`.
Trên Colab, thư mục này nằm trong runtime tạm và có thể mất khi runtime bị ngắt. Để lưu thẳng vào Google Drive, mount Drive rồi đặt `RAG_DATABASE_DIR` trước khi chạy ETL:

```python
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["RAG_DATABASE_DIR"] = "/content/drive/MyDrive/project_bio_rag/database"
```

Sau đó chạy lại lệnh ETL như bình thường:

```bash
python main.py --etl
# hoặc
python main.py --text-only
python main.py --image-only
```

Nếu PDF cũng nằm trên Drive, có thể đặt thêm:

```python
os.environ["RAG_DATA_DIR"] = "/content/drive/MyDrive/project_bio_rag/data"
```

Lưu ý: không chạy `%rm -rf database` trong notebook nếu đang muốn giữ checkpoint. Khi `RAG_DATABASE_DIR` trỏ vào Drive, muốn xóa DB thì chỉ xóa đúng thư mục Drive đó khi thật sự cần rebuild từ đầu.

## 2) Lệnh chính

| Lệnh | Mục đích |
|---|---|
| `python main.py --text-only` | ETL text (OCR + chunk + index text) |
| `python main.py --image-only` | ETL image (extract/crop + metadata + index image) |
| `python main.py --etl` | ETL full text + image |
| `python main.py --export-image-review <path.json>` | Export danh sách ảnh để reviewer chỉnh caption/xóa ảnh sai |
| `python main.py --export-image-review <path.json> --review-pdf "<file.pdf>"` | Export review theo 1 PDF |
| `python main.py --export-image-review <path.json> --review-include-completed` | Export cả item đã review |
| `python main.py --export-image-db <path.json>` | Export snapshot metadata DB (manifest) |
| `python main.py --export-image-db <path.json> --review-pdf "<file.pdf>"` | Export snapshot metadata DB theo 1 PDF |
| `python main.py --upsert-image-review-item <item.json> --review-user <name>` | Upsert 1 item metadata theo `image_id` |
| `python main.py --apply-image-review <path.json> --review-user <name>` | Apply batch review vào DB + sync image index |
| `python main.py --apply-image-review <path.json> --review-pdf "<file.pdf>" --review-user <name>` | Apply batch chỉ cho 1 PDF |
| `python main.py --replace-image-db <path.json> --review-user <name>` | Replace toàn bộ manifest + rebuild image index theo snapshot JSON |
| `python main.py --api` | Chạy Flask API server (mặc định cổng 5000) |
| `python main.py --api --port <port>` | Chạy Flask API server trên cổng tùy chỉnh |

## 3) Chạy Frontend (Next.js)

Thư mục `fe` chứa mã nguồn frontend Next.js.

### Cài đặt
```bash
cd fe
npm install
```

### Chạy cục bộ
```bash
npm run dev
```

### Chạy trên Google Colab (cùng với Backend)

1. **Chạy Backend (API)**:
   ```bash
   python main.py --api --port 5000 &
   ```

2. **Chạy Frontend**:
   ```bash
   cd fe
   npm run colab
   ```

3. **Mở giao diện**:
   Sử dụng công cụ của Colab để mở cổng 3000:
   ```python
   from google.colab import output
   output.serve_kernel_port_as_window(3000)
   ```

**Lưu ý:** Nếu bạn dùng tunnel (ngrok/localtunnel) cho Backend, hãy đặt biến môi trường `NEXT_PUBLIC_API_HOST` trỏ về URL của tunnel đó trước khi chạy Frontend.

## 4) Quy tắc apply/upsert/replace (quan trọng)

1. `--apply-image-review` là upsert theo item trong file JSON, không phải sync full theo file.
2. Xóa item khỏi array JSON không đồng nghĩa xóa item khỏi DB.
3. Muốn loại ảnh khỏi retrieval: đặt `review_status=rejected|deleted`, hoặc `is_active=false`, hoặc `delete=true`.
4. `--upsert-image-review-item`:
- `image_id` đã có -> update
- `image_id` chưa có -> create mới
5. `--replace-image-db` coi file JSON là nguồn sự thật: item không còn trong file sẽ bị xóa khỏi manifest và image index.
6. Với item thêm mới thủ công, cần đảm bảo `image_path` tồn tại để index visual hoạt động ổn định.

## 4) Flow khuyến nghị khi làm mới DB

1. Xóa DB cũ và tạo lại thư mục `database`.
2. Chạy `python main.py --text-only`.
3. Chạy `python main.py --image-only`.
4. Export review: `python main.py --export-image-review database/review_images.json`.
5. Reviewer chỉnh metadata thủ công.
6. Apply review: `python main.py --apply-image-review database/review_images.json --review-user <name>`.
7. Chạy API: `python main.py --api --port 5000`.

## 5) Ví dụ upsert 1 item

`database/one_item.json`:

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

Chạy:

```bash
python main.py --upsert-image-review-item database/one_item.json --review-user charlie
```

## 6) Ví dụ replace toàn bộ image DB

Export snapshot:

```bash
python main.py --export-image-db database/all_image_db.json
```

Chỉnh `database/all_image_db.json`, sau đó replace lại DB:

```bash
python main.py --replace-image-db database/all_image_db.json --review-user charlie
```

Payload là JSON array. Ví dụ rút gọn:

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
  }
]
```

Nếu muốn loại item khỏi retrieval nhưng vẫn giữ trong manifest snapshot, đặt:

```json
{
  "image_id": "bad_image_0001",
  "review_status": "rejected",
  "is_active": false
}
```

Nếu xóa hẳn object khỏi array rồi chạy `--replace-image-db`, item đó sẽ bị xóa khỏi manifest và image index.

## 7) Tài liệu bàn giao kỹ thuật

- Technical handover (Markdown): `document/technical_handover_rag.md`
- Technical handover (HTML): `document/technical_handover_rag.html`
- Vận hành nhanh (Markdown): `document/huong_dan_van_hanh_rag.md`
- Vận hành nhanh (HTML): `document/huong_dan_van_hanh_rag.html`
- Windows tools setup: `document/windows_tools_setup.md`

## 8) Reset tiện ích

```bash
python reset_status.py --all
python reset_status.py --image-index
python reset_status.py --images-full
python reset_status.py "D:/personal_repo/project_rag/data/SGK KHTN 6 CD.pdf"
```
