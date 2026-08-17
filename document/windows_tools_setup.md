# Hướng dẫn setup Poppler và Tesseract trên Windows

Project dùng 2 công cụ ngoài Python khi OCR PDF:

- **Poppler**: để `pdf2image` render PDF thành ảnh.
- **Tesseract OCR**: để `pytesseract` OCR tiếng Việt từ ảnh.

Nếu máy đã cài sẵn và đã thêm vào `PATH`, bạn có thể để:

```env
TESSERACT_CMD=tesseract
POPPLER_PATH=
```

Nếu chưa biết máy đã cài ở đâu, dùng 2 file zip local trong repo:

```text
D:/personal_repo/project_rag/windows_tools/poppler.zip
D:/personal_repo/project_rag/windows_tools/tesseract-ocr.zip
```

## 1) Giải nén tool local

Tạo thư mục:

```text
D:/personal_repo/project_rag/windows_tools/poppler
D:/personal_repo/project_rag/windows_tools/tesseract-ocr
```

Giải nén:

- `poppler.zip` vào `windows_tools/poppler`
- `tesseract-ocr.zip` vào `windows_tools/tesseract-ocr`

Sau khi giải nén, kiểm tra:

- Poppler: tìm thư mục chứa `pdfinfo.exe` và `pdftoppm.exe`.
- Tesseract: tìm file `tesseract.exe`.

Tùy file zip, cấu trúc thư mục có thể khác nhau. Với Poppler, path thường là:

```text
D:/personal_repo/project_rag/windows_tools/poppler/Library/bin
```

Với Tesseract, path thường là:

```text
D:/personal_repo/project_rag/windows_tools/tesseract-ocr/tesseract.exe
```

## 2) Khai báo trong `.env`

Mở file `.env` và thêm/sửa:

```env
TESSERACT_CMD=D:/personal_repo/project_rag/windows_tools/tesseract-ocr/tesseract.exe
POPPLER_PATH=D:/personal_repo/project_rag/windows_tools/poppler/Library/bin
```

Nếu thư mục sau khi giải nén khác ví dụ trên, dùng đúng path bạn tìm được:

```env
TESSERACT_CMD=<duong-dan-day-du-toi-tesseract.exe>
POPPLER_PATH=<thu-muc-chua-pdfinfo.exe-va-pdftoppm.exe>
```

## 3) Verify nhanh

Mở PowerShell tại repo:

```powershell
cd D:\personal_repo\project_rag
```

Kiểm tra Tesseract:

```powershell
& "D:\personal_repo\project_rag\windows_tools\tesseract-ocr\tesseract.exe" --version
```

Kiểm tra Poppler:

```powershell
& "D:\personal_repo\project_rag\windows_tools\poppler\Library\bin\pdfinfo.exe" -v
& "D:\personal_repo\project_rag\windows_tools\poppler\Library\bin\pdftoppm.exe" -v
```

Nếu path thực tế khác, thay path trong lệnh verify cho đúng.

## 4) Link tải khi thiếu file zip

- Poppler Windows releases: https://github.com/oschwartz10612/poppler-windows/releases/
- Hướng dẫn cài Poppler cho `pdf2image`: https://pdf2image.readthedocs.io/en/latest/installation.html
- Tesseract Windows từ UB Mannheim: https://github.com/UB-Mannheim/tesseract/wiki
- Tài liệu Tesseract Windows của UB Mannheim: https://ub-mannheim.github.io/Tesseract_Dokumentation/Tesseract_Doku_Windows.html

Sau khi tải, giải nén/cài đặt rồi cập nhật lại `TESSERACT_CMD` và `POPPLER_PATH` trong `.env`.

