"""Số liệu đánh giá truy xuất (IR) xác định cho RAG.

Mỗi câu hỏi trong bộ test được sinh ra TỪ một trang cụ thể của một cuốn sách,
nên ta biết chắc "tài liệu vàng" (ground truth) là (source_book, source_page).
Nhờ vậy có thể tính Precision@k, Recall@k (hit@k) và MRR (điểm rank) một cách
xác định bằng cách đối chiếu metadata của chunk truy xuất được — KHÔNG cần LLM,
số liệu minh bạch và lặp lại được.

Quy ước:
    - "page-level": chunk đúng khi cùng sách VÀ cùng trang nguồn (cho phép sai số
      ±PAGE_TOLERANCE trang để dung sai ranh giới chunk / OCR).
    - "book-level": chunk đúng khi chỉ cần cùng sách. Dùng để đo nhiễu chéo sách
      (cross-book contamination) — toàn bộ 12 sách nằm chung 1 collection.
"""

from typing import Callable, List, Optional

# Dung sai số trang khi đối chiếu page-level. Một câu hỏi sinh từ trang N có thể
# được trả lời bởi chunk nằm ở trang N-1/N+1 do chunk tràn qua ranh giới trang.
PAGE_TOLERANCE = 1


def _norm_source(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def make_page_relevance(src_book: str, src_page: int, tolerance: int = PAGE_TOLERANCE) -> Callable[[dict], bool]:
    """Trả về hàm kiểm tra 1 metadata chunk có khớp (sách, trang) nguồn không."""
    target_book = _norm_source(src_book)

    def is_relevant(meta: dict) -> bool:
        if _norm_source(meta.get("source")) != target_book:
            return False
        page = meta.get("page")
        if page is None or src_page is None:
            return False
        try:
            return abs(int(page) - int(src_page)) <= tolerance
        except (TypeError, ValueError):
            return False

    return is_relevant


def make_book_relevance(src_book: str) -> Callable[[dict], bool]:
    """Trả về hàm kiểm tra 1 metadata chunk có cùng sách nguồn không."""
    target_book = _norm_source(src_book)

    def is_relevant(meta: dict) -> bool:
        return _norm_source(meta.get("source")) == target_book

    return is_relevant


def precision_at_k(metas: List[dict], is_relevant: Callable[[dict], bool]) -> float:
    """Tỷ lệ chunk truy xuất được là đúng. Trả 0 nếu không truy xuất gì."""
    if not metas:
        return 0.0
    hits = sum(1 for m in metas if is_relevant(m))
    return hits / len(metas)


def recall_at_k(metas: List[dict], is_relevant: Callable[[dict], bool]) -> float:
    """Hit@k: 1.0 nếu có ÍT NHẤT một chunk đúng trong top-k, ngược lại 0.0.

    Vì mỗi câu hỏi có một trang vàng duy nhất, recall = "tìm được trang vàng hay
    không" — đây chính là hit@k, thước đo recall chuẩn cho truy xuất 1-gold.
    """
    if not metas:
        return 0.0
    return 1.0 if any(is_relevant(m) for m in metas) else 0.0


def mrr(metas: List[dict], is_relevant: Callable[[dict], bool]) -> float:
    """Mean Reciprocal Rank cho MỘT câu hỏi = 1/(thứ hạng chunk đúng đầu tiên).

    Trả 0.0 nếu không có chunk đúng nào. Lấy trung bình trên nhiều câu hỏi sẽ ra
    MRR tổng thể — "điểm rank" mà giáo viên yêu cầu.
    """
    for rank, meta in enumerate(metas, start=1):
        if is_relevant(meta):
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(metas: List[dict], src_book: str, src_page: int, k: Optional[int] = None) -> dict:
    """Tính đủ bộ số liệu page-level và book-level cho một câu hỏi.

    Args:
        metas: danh sách metadata các chunk truy xuất được, THEO THỨ TỰ hạng.
        src_book / src_page: tài liệu vàng.
        k: nếu set, chỉ xét top-k chunk đầu (mặc định lấy hết).
    """
    ranked = metas[:k] if k else metas
    page_rel = make_page_relevance(src_book, src_page)
    book_rel = make_book_relevance(src_book)
    return {
        "num_retrieved": len(ranked),
        # Page-level: thước đo nghiêm ngặt (đúng sách + đúng trang)
        "precision_page": precision_at_k(ranked, page_rel),
        "recall_page": recall_at_k(ranked, page_rel),
        "mrr_page": mrr(ranked, page_rel),
        # Book-level: đo nhiễu chéo sách (đúng sách là đủ)
        "precision_book": precision_at_k(ranked, book_rel),
        "recall_book": recall_at_k(ranked, book_rel),
        "mrr_book": mrr(ranked, book_rel),
    }
