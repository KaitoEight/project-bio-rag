# Recall@k theo từng bộ sách (top-k thô, bỏ qua relevance gate)

Chứng minh: **tăng k thì recall tăng đơn điệu** — embedding tìm được trang vàng, nút thắt nằm ở khâu cắt k/gate ở production (chỉ ~3 chunk).

| Sách | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|
| SGK KHTN 6 CD | 0.50 | 0.70 | 1.00 |
| SGK KHTN 6 CTST | 0.70 | 0.80 | 0.90 |
| SGK KHTN 6 KNTT | 0.30 | 0.50 | 0.60 |
| SGK KHTN 7 CD | 0.80 | 0.80 | 0.80 |
| SGK KHTN 7 CTST | 0.60 | 0.60 | 0.80 |
| SGK KHTN 7 KNTT | 0.70 | 0.80 | 1.00 |
| SGK KHTN 8 CTST | 0.90 | 0.90 | 0.90 |
| SGK KHTN 9 CTST | 0.70 | 0.80 | 0.80 |
| SGK KHTN8 CD | 0.60 | 0.90 | 0.90 |
| SGK KHTN8 KNTT | 0.60 | 0.70 | 0.70 |
| SGK KHTN9 CD | 0.80 | 0.80 | 0.80 |
| SGK KHTN9 KNTT | 0.70 | 0.90 | 0.90 |
| **TRUNG BÌNH** | 0.66 | 0.77 | 0.84 |