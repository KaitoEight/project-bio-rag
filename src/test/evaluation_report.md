# Báo cáo đánh giá RAG theo từng bộ sách

Tổng số bộ sách: 12 | Judge: mimo-v2.5-pro | Số câu/sách: 10

## Xếp hạng tổng thể

| Hạng | Sách | Overall | Recall@k(page) | MRR(page) | Precision(page) | Correct/5 | Faithful/5 | Relevancy/5 |
|---|---|---|---|---|---|---|---|---|
| 1 | SGK KHTN9 CD | 0.783 | 0.80 | 0.67 | 0.53 | 4.20 | 4.70 | 4.60 |
| 2 | SGK KHTN 7 CD | 0.734 | 0.80 | 0.63 | 0.43 | 3.80 | 4.40 | 4.50 |
| 3 | SGK KHTN 6 CTST | 0.716 | 0.70 | 0.65 | 0.38 | 3.80 | 4.30 | 4.70 |
| 4 | SGK KHTN 8 CTST | 0.714 | 0.80 | 0.68 | 0.50 | 3.40 | 3.80 | 4.30 |
| 5 | SGK KHTN8 KNTT | 0.710 | 0.60 | 0.50 | 0.50 | 4.00 | 4.90 | 4.40 |
| 6 | SGK KHTN8 CD | 0.692 | 0.60 | 0.53 | 0.40 | 4.00 | 4.60 | 4.50 |
| 7 | SGK KHTN 9 CTST | 0.677 | 0.70 | 0.55 | 0.33 | 4.00 | 4.10 | 4.30 |
| 8 | SGK KHTN 7 KNTT | 0.672 | 0.60 | 0.43 | 0.30 | 4.40 | 4.40 | 4.70 |
| 9 | SGK KHTN9 KNTT | 0.671 | 0.60 | 0.53 | 0.43 | 3.80 | 4.10 | 4.40 |
| 10 | SGK KHTN 7 CTST | 0.577 | 0.60 | 0.42 | 0.27 | 2.90 | 4.30 | 3.70 |
| 11 | SGK KHTN 6 CD | 0.528 | 0.50 | 0.28 | 0.27 | 3.30 | 3.40 | 3.90 |
| 12 | SGK KHTN 6 KNTT | 0.524 | 0.30 | 0.25 | 0.23 | 3.50 | 4.00 | 4.30 |

## Recall@k tăng theo k (top-k thô, bỏ qua relevance gate)

Chứng minh **tăng k thì recall tăng đơn điệu**: embedding tìm được trang vàng, nút thắt nằm ở khâu cắt k/relevance-gate ở production (chỉ ~3 chunk). So **Recall(prod)** với **Recall@10** để thấy phần recall mất đi do gate.

| Sách | Recall@3 | Recall@5 | Recall@10 | Recall(prod) |
|---|---|---|---|---|
| SGK KHTN9 CD | 0.80 | 0.80 | 0.80 | 0.80 |
| SGK KHTN 7 CD | 0.80 | 0.80 | 0.80 | 0.80 |
| SGK KHTN 6 CTST | 0.70 | 0.80 | 0.90 | 0.70 |
| SGK KHTN 8 CTST | 0.90 | 0.90 | 0.90 | 0.80 |
| SGK KHTN8 KNTT | 0.60 | 0.70 | 0.70 | 0.60 |
| SGK KHTN8 CD | 0.60 | 0.90 | 0.90 | 0.60 |
| SGK KHTN 9 CTST | 0.70 | 0.80 | 0.80 | 0.70 |
| SGK KHTN 7 KNTT | 0.70 | 0.80 | 1.00 | 0.60 |
| SGK KHTN9 KNTT | 0.70 | 0.90 | 0.90 | 0.60 |
| SGK KHTN 7 CTST | 0.60 | 0.60 | 0.80 | 0.60 |
| SGK KHTN 6 CD | 0.50 | 0.70 | 1.00 | 0.50 |
| SGK KHTN 6 KNTT | 0.30 | 0.50 | 0.60 | 0.30 |
| **TRUNG BÌNH** | 0.66 | 0.77 | 0.84 | 0.63 |

## Ghi chú số liệu
- **Recall@k(page)** = hit@k: tỷ lệ câu hỏi mà hệ truy xuất đúng trang nguồn (top-k thực tế).
- **MRR(page)** = điểm rank: trung bình 1/thứ-hạng của chunk đúng đầu tiên.
- **Precision(page)** = tỷ lệ chunk truy xuất là đúng trang nguồn.
- Cột `recall_top10_diag` trong file *_result.csv là 'trần recall' (top-10 thô) để tách lỗi gate khỏi lỗi embedding.
- **Recall@k** (top-k thô) đo khả năng embedding tìm trang vàng; **Recall(prod)** là recall sau gate (~3 chunk). Khoảng cách = recall mất do rank/cắt-k.
- **Correct/Faithful/Relevancy** do LLM thứ 2 (mimo-v2.5-pro) chấm lại câu trả lời của Qwen 2.5, thang 1-5.
- `overall_score = (retrieval_score + answer_score)/2`, dùng để xếp hạng.