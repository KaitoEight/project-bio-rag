"""Tính nhanh Recall@k (k=3/5/10) cho retrieval — KHÔNG cần Qwen, KHÔNG cần judge.

Chỉ dùng text vector DB + embedding để chứng minh: tăng k thì recall tăng. Rất nhanh
vì không nạp LLM. Đối chiếu trang vàng (source_book, source_page) trong các bộ test
đã sinh ở testsets/.

Chạy:
    python src/test/recall_at_k.py
Kết quả: in bảng + lưu testsets/../recall_at_k_report.csv (+ .md).
"""

import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from src.rag.vectorstore import VectorDB
from src.test.metrics import make_page_relevance

KS = (3, 5, 10)


def main():
    base = os.path.dirname(__file__)
    testsets = sorted(glob.glob(os.path.join(base, "testsets", "*_testset.csv")))
    if not testsets:
        print("Không tìm thấy bộ test trong testsets/. Chạy generate_testsets.py trước.")
        return

    print("Đang nạp text vector DB (MiniLM)...")
    db = VectorDB().db
    max_k = max(KS)

    rows = []
    for csv_path in testsets:
        book = os.path.basename(csv_path).replace("_testset.csv", "")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        hits = {k: 0 for k in KS}
        for _, r in df.iterrows():
            q = str(r["question"])
            is_rel = make_page_relevance(str(r["source_book"]), int(r["source_page"]))
            scored = db.similarity_search_with_score(q, k=max_k)
            ordered = [d.metadata or {} for d, _ in scored]
            for k in KS:
                if any(is_rel(m) for m in ordered[:k]):
                    hits[k] += 1
        n = len(df)
        row = {"book": book, "num_questions": n}
        row.update({f"recall@{k}": round(hits[k] / n, 4) if n else 0.0 for k in KS})
        rows.append(row)
        print(f"  {book:<22} " + "  ".join(f"R@{k}={row[f'recall@{k}']:.2f}" for k in KS))

    report = pd.DataFrame(rows)
    avg = {"book": "TRUNG BÌNH", "num_questions": int(report["num_questions"].mean())}
    avg.update({f"recall@{k}": round(report[f"recall@{k}"].mean(), 4) for k in KS})
    report = pd.concat([report, pd.DataFrame([avg])], ignore_index=True)

    out_csv = os.path.join(base, "recall_at_k_report.csv")
    report.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Markdown
    out_md = os.path.join(base, "recall_at_k_report.md")
    k_head = " | ".join(f"Recall@{k}" for k in KS)
    md = [
        "# Recall@k theo từng bộ sách (top-k thô, bỏ qua relevance gate)\n",
        "Chứng minh: **tăng k thì recall tăng đơn điệu** — embedding tìm được trang vàng, "
        f"nút thắt nằm ở khâu cắt k/gate ở production (chỉ ~3 chunk).\n",
        f"| Sách | {k_head} |",
        "|---|" + "---|" * len(KS),
    ]
    for _, r in report.iterrows():
        cells = " | ".join(f"{r[f'recall@{k}']:.2f}" for k in KS)
        name = f"**{r['book']}**" if r["book"] == "TRUNG BÌNH" else r["book"]
        md.append(f"| {name} | {cells} |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nĐã lưu: {out_csv}")
    print(f"Đã lưu: {out_md}")
    print("\n=== TRUNG BÌNH toàn bộ ===")
    print("  " + "  ".join(f"Recall@{k} = {avg[f'recall@{k}']:.3f}" for k in KS))


if __name__ == "__main__":
    main()
