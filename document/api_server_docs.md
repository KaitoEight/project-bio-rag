# Hướng Dẫn Cài Đặt và Sử Dụng Flask API Server

Tài liệu này cung cấp hướng dẫn cách khởi chạy và giao tiếp với hệ thống Biology RAG qua nền tảng API RESTful (Flask). Server này cho phép giao tiếp dễ dàng với các frontend độc lập như React hoặc Vue.

---

## 1. Cài đặt và Khởi động Server

### Cài đặt thư viện (Dependencies)
Bạn cần cài đặt đầy đủ các thư viện trong `requirements.txt` bao gồm cả `Flask` và `Flask-CORS`:

```bash
pip install -r requirements.txt
```

### Khởi động Server
Sử dụng tham số `--api` để chạy Flask web server. Bạn có thể tuỳ chọn cổng (port) bằng tham số `--port` (mặc định là 5000).

```bash
python main.py --api --port 5000
```
Sau khi khởi động, server sẽ lắng nghe ở địa chỉ: `http://0.0.0.0:5000`

> **Lưu ý:** Trong lần gọi API đầu tiên cần sử dụng model AI, server có thể mất một chút thời gian để load LLM và Embedding model vào RAM/VRAM. Các request sau đó sẽ được phản hồi cực kỳ nhanh nhờ việc các model được thiết kế chạy dưới dạng Singleton (chỉ load 1 lần duy nhất).

---

## 2. Danh Sách API Endpoints

Server hiện tại hỗ trợ các API dưới đây:

### 2.1. Upload PDF và Chạy ETL
- **Endpoint**: `POST /api/etl`
- **Mô tả**: Nhận file PDF tải lên từ client, lưu vào thư mục `DATA_DIR` và ngay lập tức chạy tiến trình ETL (tách ảnh, chunking, OCR, vector embedding) cho toàn bộ file này bằng một Background Thread.
- **Request Format**: `multipart/form-data`
  - Trường `file`: File PDF cần upload.
- **Response (202 Accepted)**:
  ```json
  {
      "message": "File <tên_file.pdf> uploaded successfully. ETL started in background.",
      "filename": "<tên_file.pdf>"
  }
  ```
- **Response Lỗi (400 Bad Request)**: Trả về khi không có file hoặc file gửi lên không phải là định dạng `.pdf`.

### 2.2. Kiểm tra tiến trình ETL (Polling)
- **Endpoint**: `GET /api/etl/status`
- **Mô tả**: Do tiến trình ETL mất rất nhiều thời gian, Frontend nên sử dụng cơ chế Polling (gọi định kỳ mỗi vài giây) API này để biết lúc nào file PDF đã được xử lý xong.
- **Query Params**: 
  - `filename`: Tên file PDF muốn kiểm tra (Ví dụ: `?filename=sach_sinh_hoc_10.pdf`).
- **Response (200 OK)**:
  ```json
  {
      "status": "processing" | "completed" | "error",
      "message": "Chi tiết trạng thái hiện tại"
  }
  ```
- **Response (404 Not Found)**:
  ```json
  {
      "status": "not_found",
      "message": "No ETL task found for this file."
  }
  ```

### 2.3. RAG Chat & Tìm kiếm Ảnh
- **Endpoint**: `POST /api/chat`
- **Mô tả**: Điểm kết nối chính để trò chuyện với trợ lý ảo Biology RAG. Hàm sẽ sử dụng phương pháp Hybrid Retrieval (tìm kiếm text kết hợp hình ảnh) và trả về câu trả lời tổng hợp.
- **Request Format**: `application/json`
  ```json
  {
      "question": "Thành phần của một tế bào động vật bao gồm những gì?"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
      "answer": "Câu trả lời chi tiết do LLM sinh ra \n\n📚 Thông tin được tham khảo từ: Trang X - Sách Giáo Khoa",
      "images": [
          {
              "image_path": "C:\\path\\to\\project_rag\\database\\images\\...\\image_1.jpg",
              "label": "Mô tả hình ảnh (Trang X, sach_sinh_hoc_10.pdf)",
              "metadata": { ... toàn bộ metadata của ảnh ... }
          }
      ]
  }
  ```
> **Lưu ý với API `/api/chat`**: Đường dẫn trả về của ảnh (`image_path`) là đường dẫn vật lý (absolute path) trên ổ cứng máy chủ. Nếu Frontend của bạn là ứng dụng Local/Electron, nó có thể trực tiếp lấy ảnh bằng đường dẫn này. Nếu Frontend tách biệt (vd: website), bạn cần tự cấu hình route để serve static files hoặc yêu cầu backend cung cấp 1 API trả về data ảnh/stream.

### 2.4. RAG Chat dạng stream
- **Endpoint**: `POST /api/chat/stream`
- **Cách khác**: gọi `POST /api/chat` với body có thêm `"stream": true`.
- **Mô tả**: Trả câu trả lời dạng `text/event-stream` để frontend render từng phần như ChatGPT. Backend gửi các event:
  - `status`: trạng thái như `retrieving` hoặc `answering`.
  - `answer_delta`: một đoạn text mới của câu trả lời.
  - `done`: câu trả lời đã clean cuối cùng và danh sách ảnh liên quan.
  - `error`: lỗi nếu quá trình stream thất bại.

Ví dụ request:
```json
{
    "question": "Thành phần của một tế bào động vật bao gồm những gì?"
}
```

Ví dụ frontend dùng `fetch`:
```js
const response = await fetch("http://localhost:5000/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder("utf-8");
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const events = buffer.split("\n\n");
  buffer = events.pop() || "";

  for (const rawEvent of events) {
    const eventName = rawEvent.match(/^event: (.+)$/m)?.[1];
    const dataLine = rawEvent.match(/^data: (.+)$/m)?.[1];
    if (!eventName || !dataLine) continue;

    const payload = JSON.parse(dataLine);
    if (eventName === "answer_delta") {
      appendToAnswer(payload.delta);
    }
    if (eventName === "done") {
      replaceAnswer(payload.answer);
      renderImages(payload.images);
    }
  }
}
```

### 2.5. Lấy dữ liệu Image Review (Snapshot)
- **Endpoint**: `GET /api/images`
- **Mô tả**: Truy xuất toàn bộ dữ liệu metadata của ảnh có trong Database (file `image_review_manifest.jsonl`). Thích hợp để render dữ liệu ra giao diện Bảng (Table) trên FE cho tác vụ Review/Edit Ảnh.
- **Query Params (Tùy chọn)**:
  - `pdf_filename`: Lọc kết quả metadata chỉ thuộc về một cuốn sách cụ thể.
- **Response (200 OK)**: Trả về một mảng JSON array chứa đầy đủ metadata cho từng ảnh.

### 2.6. Thay thế & Cập nhật dữ liệu Image Review
- **Endpoint**: `PUT /api/images` hoặc `POST /api/images`
- **Mô tả**: Nhận một mảng dữ liệu ảnh đã qua chỉnh sửa từ Frontend (ví dụ: đã sửa caption, xoá ảnh sai sót, sửa keyword,...). Sau đó cập nhật cấu hình nội bộ và đồng bộ tự động lên Vector Database của Hình ảnh.
- **Query Params (Tùy chọn)**:
  - `reviewed_by`: Tên người review (mặc định là `react-frontend`).
- **Request Format**: `application/json` (Danh sách các Object)
  ```json
  [
      {
          "image_id": "ab12cd34",
          "review_status": "approved",
          "caption_vi_manual": "Đây là chú thích ảnh tôi vừa chỉnh tay",
          "delete": false
      },
      ...
  ]
  ```
- **Response (200 OK)**:
  ```json
  {
      "replaced": 150,
      "removed": 2,
      "inactive": 5,
      "upserted": 145,
      "skipped": 0
  }
  ```
