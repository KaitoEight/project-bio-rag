"""Đánh giá hệ RAG (Qwen 2.5) theo từng bộ sách + LLM thứ 2 (Gemini) chấm lại.

Luồng cho mỗi câu hỏi trong từng bộ test:
    1. TRUY XUẤT: gọi đúng pipeline thật (HybridRetriever) -> lấy chunk text kèm
       metadata (source, page).
    2. SỐ LIỆU IR (xác định): đối chiếu metadata với (source_book, source_page)
       của câu hỏi -> Precision@k, Recall@k (hit@k), MRR (điểm rank).
       Tính cả mức page-level (nghiêm) và book-level (đo nhiễu chéo sách).
    3. SINH CÂU TRẢ LỜI: ghép context -> prompt -> Qwen 2.5 -> parser (y như API).
    4. LLM THỨ 2 CHẤM LẠI: Gemini 2.5 Flash chấm câu trả lời của Qwen theo
       correctness / faithfulness / relevancy (1-5) so với đáp án chuẩn + context.

Kết quả:
    - testsets/<book>_result.csv  : chi tiết từng câu.
    - evaluation_report.csv       : tổng hợp + XẾP HẠNG 12 sách.
    - evaluation_report.md        : bảng leaderboard dễ đọc.

Chạy (sau khi đã có testsets):
    python src/test/evaluator.py
"""

import os
import sys
import glob
import json
import re
import logging

# Cho phép import package `src`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
from dotenv import load_dotenv

from src.app.dependencies import AppServices
from src.test.metrics import evaluate_retrieval
from src.test.eval_llm import get_eval_llm, is_configured, config_help

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_MODEL = os.getenv("EVAL_LLM_MODEL", "(chưa cấu hình)")
# Các mức k cho recall "thô" (top-k similarity, BỎ QUA relevance gate) — dùng để
# chứng minh: tăng k thì recall tăng. recall@<max> chính là "trần recall".
RAW_RECALL_KS = (3, 5, 10)

JUDGE_PROMPT = """Bạn là giám khảo chấm chất lượng câu trả lời của một trợ lý AI môn
Khoa học tự nhiên. Hãy chấm KHÁCH QUAN, dựa trên đáp án chuẩn và ngữ cảnh.

[CÂU HỎI]:
{question}

[ĐÁP ÁN CHUẨN (ground truth)]:
{ground_truth}

[NGỮ CẢNH HỆ THỐNG TRUY XUẤT ĐƯỢC]:
{context}

[CÂU TRẢ LỜI CỦA TRỢ LÝ (cần chấm)]:
{answer}

Chấm theo 3 tiêu chí, thang điểm 1-5 (5 là tốt nhất):
- correctness: câu trả lời có ĐÚNG so với đáp án chuẩn không.
- faithfulness: câu trả lời có BÁM SÁT ngữ cảnh, không bịa (hallucination) không.
- relevancy: có trả lời ĐÚNG TRỌNG TÂM câu hỏi không.

CHỈ trả JSON thuần (không markdown):
{{"correctness": <1-5>, "faithfulness": <1-5>, "relevancy": <1-5>, "reasoning": "<ngắn gọn>"}}
"""


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def get_answer_and_context(question: str) -> dict:
    """Gọi hệ RAG Qwen 2.5 thật. Trả về answer, contexts, và metadata chunk.

    Tái sử dụng đúng pipeline của API /api/chat:
    HybridRetriever -> ghép context -> prompt -> Qwen 2.5 -> parser.
    """
    services = AppServices.get_instance()

    result = services.hybrid_retriever.search(question)
    text_docs = result.text_docs

    contexts = [d.page_content for d in text_docs if getattr(d, "page_content", None)]
    metas = [getattr(d, "metadata", {}) or {} for d in text_docs]

    if not contexts:
        answer = (
            "Mình tìm thấy hình ảnh liên quan trong cơ sở dữ liệu ảnh."
            if result.image_only_query
            else "Thông tin này không được đề cập trong sách giáo khoa."
        )
        return {"answer": answer, "contexts": contexts, "metas": metas}

    context_str = "\n\n".join(contexts)
    prompt = services.rag.prompt.format(context=context_str, question=question)
    try:
        raw = services.llm.invoke(prompt)
        answer = services.rag.answer_parser.parse(raw)
    except Exception as exc:
        logger.error("Qwen lỗi cho câu hỏi %r: %s", question, exc)
        answer = "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."

    return {"answer": answer, "contexts": contexts, "metas": metas}


def raw_recall_at_ks(question: str, src_book: str, src_page: int, ks=RAW_RECALL_KS) -> dict:
    """Recall@k page-level cho nhiều mức k, dùng top-k similarity THÔ (bỏ qua gate).

    Lấy top-max(ks) một lần rồi cắt tiền tố để tính hit@3/5/10 → cho thấy recall
    tăng đơn điệu theo k. recall@<max> chính là 'trần recall' (embedding có tìm
    ra trang vàng không), tách bạch với recall production (bị gate cắt còn ~3).
    """
    from src.test.metrics import make_page_relevance
    services = AppServices.get_instance()
    max_k = max(ks)
    try:
        scored = services.hybrid_retriever.text_db.db.similarity_search_with_score(question, k=max_k)
    except Exception as exc:
        logger.warning("Raw recall lỗi: %s", exc)
        return {f"recall@{k}_raw": float("nan") for k in ks}

    is_rel = make_page_relevance(src_book, src_page)
    ordered_metas = [d.metadata or {} for d, _ in scored]
    out = {}
    for k in ks:
        out[f"recall@{k}_raw"] = 1.0 if any(is_rel(m) for m in ordered_metas[:k]) else 0.0
    return out


def judge_answer(judge_llm, question: str, ground_truth: str, context: str, answer: str) -> dict:
    try:
        resp = judge_llm.invoke(JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth,
            context=(context or "(không có)")[:6000], answer=answer,
        ))
        data = _parse_json(resp.content if hasattr(resp, "content") else str(resp))
        return {
            "judge_correctness": float(data.get("correctness", 0)),
            "judge_faithfulness": float(data.get("faithfulness", 0)),
            "judge_relevancy": float(data.get("relevancy", 0)),
            "judge_reasoning": str(data.get("reasoning", "")).strip(),
        }
    except Exception as exc:
        logger.warning("Judge lỗi: %s", exc)
        return {"judge_correctness": float("nan"), "judge_faithfulness": float("nan"),
                "judge_relevancy": float("nan"), "judge_reasoning": f"LỖI: {exc}"}


def evaluate_book(csv_path: str, judge_llm) -> dict:
    df = pd.read_csv(csv_path)
    book = os.path.basename(csv_path).replace("_testset.csv", "")
    print(f"\n=== Đánh giá: {book} ({len(df)} câu) ===")

    records = []
    for i, row in df.iterrows():
        q = str(row["question"])
        gt = str(row.get("ground_truth", ""))
        src_book = str(row["source_book"])
        src_page = int(row["source_page"])

        rag = get_answer_and_context(q)
        ir = evaluate_retrieval(rag["metas"], src_book, src_page)
        raw_recalls = raw_recall_at_ks(q, src_book, src_page)
        verdict = judge_answer(judge_llm, q, gt, "\n\n".join(rag["contexts"]), rag["answer"])

        retrieved_sources = "; ".join(
            f"{(m.get('source') or '?')}:p{m.get('page')}" for m in rag["metas"]
        )
        records.append({
            "question": q,
            "source_book": src_book,
            "source_page": src_page,
            **ir,
            **raw_recalls,
            "retrieved": retrieved_sources,
            "rag_answer": rag["answer"],
            "ground_truth": gt,
            **verdict,
        })
        print(f"  [{i + 1:>2}/{len(df)}] P_page={ir['precision_page']:.2f} "
              f"R_page={ir['recall_page']:.0f} MRR={ir['mrr_page']:.2f} "
              f"correct={verdict['judge_correctness']:.0f}/5")

    res_df = pd.DataFrame(records)
    res_df.to_csv(csv_path.replace("_testset.csv", "_result.csv"), index=False, encoding="utf-8-sig")

    num_cols = [
        "precision_page", "recall_page", "mrr_page",
        "precision_book", "recall_book", "mrr_book",
        *[f"recall@{k}_raw" for k in RAW_RECALL_KS],
        "judge_correctness", "judge_faithfulness", "judge_relevancy",
    ]
    summary = {"book": book, "num_questions": len(res_df)}
    summary.update({c: round(res_df[c].mean(skipna=True), 4) for c in num_cols})
    return summary


def run_evaluation(testsets_dir: str, report_csv: str, report_md: str):
    judge_llm = get_eval_llm(temperature=0.0)

    csv_files = sorted(glob.glob(os.path.join(testsets_dir, "*_testset.csv")))
    if not csv_files:
        print("Không tìm thấy bộ test. Chạy generate_testsets.py trước.")
        return

    summaries = []
    for csv_path in csv_files:
        try:
            summaries.append(evaluate_book(csv_path, judge_llm))
        except Exception as exc:
            import traceback
            print(f"Lỗi khi đánh giá {csv_path}: {exc}")
            traceback.print_exc()

    if not summaries:
        return

    report = pd.DataFrame(summaries)
    # Điểm tổng hợp để XẾP HẠNG: 50% chất lượng truy xuất + 50% chất lượng trả lời.
    report["retrieval_score"] = report[["recall_page", "mrr_page", "precision_page"]].mean(axis=1)
    report["answer_score"] = (
        report[["judge_correctness", "judge_faithfulness", "judge_relevancy"]].mean(axis=1) / 5.0
    )
    report["overall_score"] = (report["retrieval_score"] + report["answer_score"]) / 2.0
    report = report.sort_values("overall_score", ascending=False).reset_index(drop=True)
    report.insert(0, "rank", report.index + 1)
    report.to_csv(report_csv, index=False, encoding="utf-8-sig")

    # Bảng leaderboard markdown.
    lines = [
        "# Báo cáo đánh giá RAG theo từng bộ sách\n",
        f"Tổng số bộ sách: {len(report)} | Judge: {os.getenv('EVAL_LLM_MODEL', '?')} | "
        f"Số câu/sách: {int(report['num_questions'].mean())}\n",
        "## Xếp hạng tổng thể\n",
        "| Hạng | Sách | Overall | Recall@k(page) | MRR(page) | Precision(page) | "
        "Correct/5 | Faithful/5 | Relevancy/5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in report.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['book']} | {r['overall_score']:.3f} | "
            f"{r['recall_page']:.2f} | {r['mrr_page']:.2f} | {r['precision_page']:.2f} | "
            f"{r['judge_correctness']:.2f} | {r['judge_faithfulness']:.2f} | {r['judge_relevancy']:.2f} |"
        )

    # Bảng recall@k: chứng minh tăng k thì recall tăng (top-k thô, bỏ qua gate).
    raw_cols = [f"recall@{k}_raw" for k in RAW_RECALL_KS]
    k_headers = " | ".join(f"Recall@{k}" for k in RAW_RECALL_KS)
    lines += [
        "\n## Recall@k tăng theo k (top-k thô, bỏ qua relevance gate)\n",
        "Cho thấy embedding tìm được trang vàng ở mức nào; tăng k thì recall tăng đơn điệu. "
        f"Recall@{max(RAW_RECALL_KS)} là 'trần recall'. So với **Recall(prod)** (chỉ ~3 chunk sau gate) "
        "để thấy nút thắt nằm ở khâu gate/cắt-k, không phải embedding.\n",
        f"| Sách | {k_headers} | Recall(prod) |",
        "|---|" + "---|" * (len(RAW_RECALL_KS) + 1),
    ]
    for _, r in report.iterrows():
        cells = " | ".join(f"{r[c]:.2f}" for c in raw_cols)
        lines.append(f"| {r['book']} | {cells} | {r['recall_page']:.2f} |")
    # Dòng trung bình toàn bộ sách.
    avg_cells = " | ".join(f"{report[c].mean():.2f}" for c in raw_cols)
    lines.append(f"| **TRUNG BÌNH** | {avg_cells} | {report['recall_page'].mean():.2f} |")

    lines += [
        "\n## Ghi chú số liệu",
        "- **Recall@k(page)** = hit@k: tỷ lệ câu hỏi mà hệ truy xuất đúng trang nguồn (top-k thực tế).",
        "- **MRR(page)** = điểm rank: trung bình 1/thứ-hạng của chunk đúng đầu tiên.",
        "- **Precision(page)** = tỷ lệ chunk truy xuất là đúng trang nguồn.",
        f"- **Recall@{RAW_RECALL_KS}** (top-k thô) đo khả năng embedding tìm thấy trang vàng; "
        "**Recall(prod)** là recall thực tế sau relevance gate (~3 chunk). Khoảng cách giữa hai cái "
        "định lượng phần recall mất đi do khâu rank/cắt-k.",
        "- **Correct/Faithful/Relevancy** do LLM thứ 2 chấm lại câu trả lời của Qwen 2.5, thang 1-5.",
        "- `overall_score = (retrieval_score + answer_score)/2`, dùng để xếp hạng.",
    ]
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nĐã lưu báo cáo: {report_csv}")
    print(f"Đã lưu leaderboard: {report_md}")
    print("\n" + report[["rank", "book", "overall_score", "recall_page",
                          "mrr_page", "judge_correctness"]].to_string(index=False))


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    TESTSETS_DIR = os.path.join(base, "testsets")
    REPORT_CSV = os.path.join(base, "evaluation_report.csv")
    REPORT_MD = os.path.join(base, "evaluation_report.md")

    if not is_configured():
        print(config_help())
    else:
        run_evaluation(TESTSETS_DIR, REPORT_CSV, REPORT_MD)
