"""Flask API Server for Biology RAG"""

import json
import logging
import os
import threading
import torch
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from transformers import TextIteratorStreamer
from werkzeug.utils import secure_filename

from src.config import DATA_DIR, IMAGES_DIR, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_TOP_P
from src.app.dependencies import AppServices
from src.etl.image_review import ImageReviewManager

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# In-memory status tracker for ETL polling
etl_status = {}


def sse_event(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def build_gallery_items(image_docs):
    gallery_items = []
    for doc in image_docs:
        if hasattr(doc, "metadata"):
            image_path = doc.metadata.get("image_path", "")
            page = doc.metadata.get("page_number", "?")
            pdf = doc.metadata.get("pdf_filename", "Sách Giáo Khoa")

            caption = doc.metadata.get("caption") or ""
            figure_caption = doc.metadata.get("figure_caption") or ""
            figure_label = doc.metadata.get("figure_label") or ""
            context_text = doc.metadata.get("context_text") or ""
            fallback_text = (doc.page_content or "").splitlines()[0] if doc.page_content else ""
            label_text = figure_caption or figure_label or caption or context_text or fallback_text

            label = f"{label_text[:80]}... (Trang {page}, {pdf})" if label_text else f"Trang {page} - {pdf}"

            gallery_items.append({
                "image_path": image_path,
                "label": label,
                "metadata": doc.metadata
            })
    return gallery_items


def prepare_chat_payload(question, top_k=None):
    services = AppServices.get_instance()
    result = services.hybrid_retriever.search(question, text_k=top_k)
    text_docs = result.text_docs
    image_docs = result.image_docs
    image_only_query = result.image_only_query
    gallery_items = build_gallery_items(image_docs)

    if not text_docs and not image_docs:
        if image_only_query:
            return {
                "mode": "static",
                "answer": "Không tìm thấy hình ảnh liên quan trong cơ sở dữ liệu ảnh.",
                "images": gallery_items,
                "services": services,
            }
        return {
            "mode": "static",
            "answer": "Hệ thống chưa tìm thấy tài liệu nào liên quan đến câu hỏi này.",
            "images": gallery_items,
            "services": services,
        }

    if image_only_query:
        return {
            "mode": "static",
            "answer": f"Mình tìm thấy {len(image_docs)} hình ảnh liên quan trong cơ sở dữ liệu ảnh.",
            "images": gallery_items,
            "services": services,
        }

    if not text_docs:
        return {
            "mode": "static",
            "answer": "Không tìm thấy thông tin dạng văn bản liên quan. Vui lòng thử câu hỏi khác.",
            "images": gallery_items,
            "services": services,
        }

    context_texts = [doc.page_content for doc in text_docs if hasattr(doc, "page_content")]
    context_str = "\n\n".join(context_texts)

    citations = set()
    for doc in text_docs:
        source = doc.metadata.get("source", "Sách Giáo Khoa") if hasattr(doc, "metadata") else "Sách Giáo Khoa"
        page = doc.metadata.get("page", "?") if hasattr(doc, "metadata") else "?"
        citations.add(f"Trang {page} - {source}")
    citations_str = " | ".join(sorted(citations))

    return {
        "mode": "llm",
        "formatted_prompt": services.rag.prompt.format(context=context_str, question=question),
        "citations_str": citations_str,
        "images": gallery_items,
        "services": services,
    }


def append_citations(answer, citations_str):
    if "không được đề cập" not in answer.lower() and citations_str:
        return f"{answer}\n\n📚 Thông tin được tham khảo từ: {citations_str}"
    return answer


def stream_static_answer(answer):
    for word in answer.split(" "):
        yield f"{word} "


def stream_llm_text(services, formatted_prompt):
    model_pipeline = getattr(services.llm, "pipeline", None)
    if model_pipeline is None:
        yield str(services.llm.invoke(formatted_prompt))
        return

    tokenizer = model_pipeline.tokenizer
    model = model_pipeline.model
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": LLM_MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    generation_error = {}

    def generate():
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except Exception as exc:
            generation_error["error"] = exc
            if hasattr(streamer, "on_finalized_text"):
                streamer.on_finalized_text("", stream_end=True)

    thread = threading.Thread(target=generate)
    thread.start()
    for text in streamer:
        if text:
            yield text
    thread.join()

    if generation_error:
        raise generation_error["error"]


def create_chat_stream_response(question):
    def generate_events():
        try:
            yield sse_event("status", {"type": "status", "stage": "retrieving"})
            payload = prepare_chat_payload(question, top_k=data.get("top_k"))

            if payload["mode"] == "static":
                answer = payload["answer"]
                yield sse_event("status", {"type": "status", "stage": "answering"})
                for chunk in stream_static_answer(answer):
                    yield sse_event("answer_delta", {"type": "answer_delta", "delta": chunk})
                yield sse_event("done", {
                    "type": "done",
                    "answer": answer,
                    "images": payload["images"],
                })
                return

            yield sse_event("status", {"type": "status", "stage": "answering"})
            chunks = []
            for chunk in stream_llm_text(payload["services"], payload["formatted_prompt"]):
                chunks.append(chunk)
                yield sse_event("answer_delta", {"type": "answer_delta", "delta": chunk})

            parsed_answer = payload["services"].rag.answer_parser.parse("".join(chunks))
            answer = append_citations(parsed_answer, payload["citations_str"])
            yield sse_event("done", {
                "type": "done",
                "answer": answer,
                "images": payload["images"],
            })
        except Exception as e:
            logger.error(f"Error streaming answer: {e}", exc_info=True)
            yield sse_event("error", {"type": "error", "error": str(e)})

    return Response(
        stream_with_context(generate_events()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

def run_etl_background(filename):
    from main import run_etl  # Import locally to avoid circular dependency
    etl_status[filename] = {"status": "processing", "message": "ETL pipeline is running"}
    try:
        run_etl()
        etl_status[filename] = {"status": "completed", "message": "ETL completed successfully"}
    except Exception as e:
        logger.error(f"ETL failed for {filename}: {e}")
        etl_status[filename] = {"status": "error", "message": str(e)}

@app.route('/api/etl', methods=['POST'])
def upload_and_etl():
    """Upload a PDF and trigger ETL process."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, filename)
        file.save(filepath)
        
        # Start ETL in background
        thread = threading.Thread(target=run_etl_background, args=(filename,))
        thread.start()
        
        return jsonify({"message": f"File {filename} uploaded successfully. ETL started in background.", "filename": filename}), 202
    return jsonify({"error": "Only PDF files are allowed"}), 400

@app.route('/api/etl/status', methods=['GET'])
def get_etl_status():
    """Check ETL status for a given PDF filename."""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "Filename parameter is required"}), 400
    
    status_info = etl_status.get(filename)
    if not status_info:
        # Check if it was already processed before server start
        try:
            from main import get_processed_files, get_processed_images
            text_done = filename in get_processed_files()
            image_done = filename in get_processed_images()
            if text_done and image_done:
                return jsonify({"status": "completed", "message": "File was already processed."}), 200
        except Exception as e:
            logger.warning(f"Failed to check processed files: {e}")
            
        return jsonify({"status": "not_found", "message": "No ETL task found for this file."}), 404
        
    return jsonify(status_info), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint using book_rag with Gemini."""
    import asyncio
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400

    question = data['question']
    try:
        # Try book_rag first (Gemini + vectorstore)
        import sys as _sys, os as _os
        _sys.path.insert(0, str(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))))
        import importlib.util as _ius
        _spec = _ius.spec_from_file_location(
            "book_rag",
            _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                         "src/rag/book_rag.py"))
        _mod = _ius.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        answer = loop.run_until_complete(_mod.answer_question(question))
        loop.close()
        return jsonify({"answer": answer, "images": []})
    except Exception as e:
        logger.error(f"book_rag failed: {e}")
        # Fallback to old RAG chain
        try:
            payload = prepare_chat_payload(question, top_k=data.get("top_k"))
            if payload["mode"] == "static":
                answer = payload["answer"]
            else:
                try:
                    llm_response = payload["services"].llm.invoke(payload["formatted_prompt"])
                    answer = payload["services"].rag.answer_parser.parse(llm_response)
                except Exception as llm_err:
                    logger.error(f"RAG chain failed: {llm_err}")
                    answer = "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."
                answer = append_citations(answer, payload["citations_str"])
            return jsonify({"answer": answer, "images": payload["images"]})
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}", exc_info=True)
            return jsonify({"error": str(e2)}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Stream chat answer chunks through Server-Sent Events."""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400

    return create_chat_stream_response(data['question'])

@app.route('/api/images', methods=['GET'])
def get_images():
    """Retrieve full image database snapshot."""
    pdf_filename = request.args.get('pdf_filename')
    manager = ImageReviewManager()
    try:
        snapshot = manager.get_db_snapshot(pdf_filename=pdf_filename)
        return jsonify(snapshot), 200
    except Exception as e:
        logger.error(f"Error fetching images: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/images', methods=['PUT', 'POST'])
def update_images():
    """Replace image database from a JSON array payload."""
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify({"error": "Payload must be a JSON array"}), 400
        
    reviewed_by = request.args.get('reviewed_by', 'react-frontend')
    manager = ImageReviewManager()
    
    try:
        summary = manager.replace_image_db_from_payload(payload=data, reviewed_by=reviewed_by)
        return jsonify(summary), 200
    except Exception as e:
        logger.error(f"Error replacing image db: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve static images from the database."""
    return send_from_directory(str(IMAGES_DIR), filename)

def run_api(host='0.0.0.0', port=5000):
    logger.info("Initializing AppServices before starting Flask...")
    AppServices.get_instance()
    logger.info(f"Starting Flask API server on {host}:{port}...")
    app.run(host=host, port=port, debug=False)
