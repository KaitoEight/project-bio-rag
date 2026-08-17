"""Main entry point for Biology RAG system."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from src.config import LOG_LEVEL, DATA_DIR, PERSIST_DIR, IMAGES_DIR, PROCESSED_FILES_LOG, PROCESSED_IMAGES_LOG

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_processed_files():
    """Read list of processed files from checkpoint log."""
    if not os.path.exists(PROCESSED_FILES_LOG):
        return set()
    with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mark_file_as_processed(filename: str):
    """Mark a file as processed in checkpoint log."""
    with open(PROCESSED_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")


def get_processed_images():
    """Read list of processed files for images from checkpoint log."""
    if not os.path.exists(PROCESSED_IMAGES_LOG):
        return set()
    with open(PROCESSED_IMAGES_LOG, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mark_image_as_processed(filename: str):
    """Mark a file as processed for images in checkpoint log."""
    with open(PROCESSED_IMAGES_LOG, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")


def run_etl_text_only(ocr_strategy: str = "hybrid"):
    """Run ETL pipeline for text only: PDF loading, OCR, chunking, and storing to ChromaDB."""
    logger.info(f"Starting ETL pipeline (TEXT ONLY, strategy={ocr_strategy})...")

    os.makedirs(PERSIST_DIR, exist_ok=True)

    from tqdm import tqdm
    from src.etl import HybridOCRLoader, TextSplitter, ProcessingStatus, compute_file_hash

    loader = HybridOCRLoader(strategy=ocr_strategy)
    text_splitter = TextSplitter()
    status_tracker = ProcessingStatus()

    from src.rag.vectorstore import VectorDB

    text_vdb = VectorDB()
    text_db = text_vdb.db

    import glob

    pdf_files = glob.glob(f"{DATA_DIR}/*.pdf")

    if not pdf_files:
        logger.error(f"No PDF files found in {DATA_DIR}")
        return

    logger.info(f"Total PDFs in directory: {len(pdf_files)}")

    processed_files = get_processed_files()
    logger.info(f"Previously processed files: {len(processed_files)}")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs for text"):
        filename = os.path.basename(pdf_file)
        if filename in processed_files:
            logger.info(f"[{filename}] Already processed for text, skipping")
            continue

        logger.info(f"Processing: {filename}")

        try:
            pdf_hash = compute_file_hash(pdf_file)

            docs = loader.load_pdf(pdf_file)
            if not docs:
                logger.warning(f"No text extracted from {filename}")
                continue

            ocr_text_per_page = {doc.metadata.get(
                "page", i + 1): doc.page_content for i, doc in enumerate(docs)}

            pages_to_index = status_tracker.get_pages_needing_text(
                pdf_hash, len(docs))
            if not pages_to_index:
                logger.info(
                    f"[{filename}] All pages already indexed, skipping")
                mark_file_as_processed(filename)
                continue

            logger.info(f"[{filename}] Pages to index: {pages_to_index}")

            docs_to_add = [doc for doc in docs if doc.metadata.get(
                "page") in pages_to_index]

            if docs_to_add:
                split_docs = text_splitter.split(docs_to_add)
                logger.info(f"Split into {len(split_docs)} chunks")
                text_db.add_documents(split_docs)

                for page_num in pages_to_index:
                    status_tracker.mark_text_indexed(
                        pdf_hash, page_num, filename)

            mark_file_as_processed(filename)
            logger.info(f"Completed: {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info("ETL (TEXT) pipeline completed!")


def run_etl_image_only():
    """Run ETL pipeline for images only: extract, filter, caption, and store images."""
    logger.info("Starting ETL pipeline (IMAGE ONLY)...")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    from tqdm import tqdm
    from src.etl import RobustOCRLoader, ImageProcessor, ProcessingStatus, compute_file_hash

    loader = RobustOCRLoader()
    image_processor = ImageProcessor()
    status_tracker = ProcessingStatus()

    from src.rag.image_vectorstore import ImageVectorDB

    image_vdb = ImageVectorDB()

    import glob

    pdf_files = glob.glob(f"{DATA_DIR}/*.pdf")

    if not pdf_files:
        logger.error(f"No PDF files found in {DATA_DIR}")
        return

    logger.info(f"Total PDFs in directory: {len(pdf_files)}")

    processed_images = get_processed_images()
    logger.info(f"Previously processed files for images: {len(processed_images)}")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs for images"):
        filename = os.path.basename(pdf_file)
        if filename in processed_images:
            logger.info(f"[{filename}] Already processed for images, skipping")
            continue

        logger.info(f"Processing: {filename}")

        try:
            pdf_hash = compute_file_hash(pdf_file)

            docs = loader.load_pdf(pdf_file)
            if not docs:
                logger.warning(f"No text extracted from {filename}")
                continue

            ocr_text_per_page = {doc.metadata.get(
                "page", i + 1): doc.page_content for i, doc in enumerate(docs)}

            pages_to_process = [
                page_num
                for page_num in range(1, len(docs) + 1)
                if status_tracker.needs_image_processing_versioned(
                    pdf_hash,
                    page_num,
                    required_version=image_processor.image_extraction_version,
                )
            ]
            if not pages_to_process:
                logger.info(
                    f"[{filename}] All pages already processed for images, skipping")
                continue

            logger.info(
                f"[{filename}] Pages to extract images: {pages_to_process}")

            image_docs = image_processor.extract_images_from_pdf(
                pdf_path=pdf_file,
                pdf_hash=pdf_hash,
                pdf_filename=filename,
                ocr_text_per_page=ocr_text_per_page,
            )

            if image_docs:
                image_vdb.add_documents(image_docs)
                logger.info(
                    f"[{filename}] Added {len(image_docs)} images to ImageVectorDB")

            mark_image_as_processed(filename)
            logger.info(f"Completed images for: {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info("ETL (IMAGE) pipeline completed!")


def run_etl(ocr_strategy: str = "hybrid"):
    """Run full ETL pipeline: both text and images."""
    logger.info(f"Starting ETL pipeline (FULL, strategy={ocr_strategy})...")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    from tqdm import tqdm
    from src.etl import HybridOCRLoader, TextSplitter, ImageProcessor, ProcessingStatus, compute_file_hash

    loader = HybridOCRLoader(strategy=ocr_strategy)
    text_splitter = TextSplitter()
    image_processor = ImageProcessor()
    status_tracker = ProcessingStatus()

    from src.rag.vectorstore import VectorDB
    from src.rag.image_vectorstore import ImageVectorDB

    text_vdb = VectorDB()
    image_vdb = ImageVectorDB()

    import glob

    pdf_files = glob.glob(f"{DATA_DIR}/*.pdf")

    if not pdf_files:
        logger.error(f"No PDF files found in {DATA_DIR}")
        return

    logger.info(f"Total PDFs in directory: {len(pdf_files)}")

    processed_files = get_processed_files()
    processed_images = get_processed_images()

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        filename = os.path.basename(pdf_file)
        text_done = filename in processed_files
        image_done = filename in processed_images

        if text_done and image_done:
            logger.info(f"[{filename}] Already processed for both text and images, skipping")
            continue

        logger.info(f"Processing: {filename}")

        try:
            pdf_hash = compute_file_hash(pdf_file)

            docs = loader.load_pdf(pdf_file)
            if not docs:
                logger.warning(f"No text extracted from {filename}")
                continue

            ocr_text_per_page = {doc.metadata.get(
                "page", i + 1): doc.page_content for i, doc in enumerate(docs)}

            if not text_done:
                pages_needing_text = status_tracker.get_pages_needing_text(
                    pdf_hash, len(docs))
                if pages_needing_text:
                    logger.info(
                        f"[{filename}] Indexing {len(pages_needing_text)} pages for text")
                    docs_to_add = [doc for doc in docs if doc.metadata.get(
                        "page") in pages_needing_text]
                    if docs_to_add:
                        split_docs = text_splitter.split(docs_to_add)
                        text_vdb.db.add_documents(split_docs)
                        for page_num in pages_needing_text:
                            status_tracker.mark_text_indexed(
                                pdf_hash, page_num, filename)

            if not image_done:
                pages_needing_images = [
                    page_num
                    for page_num in range(1, len(docs) + 1)
                    if status_tracker.needs_image_processing_versioned(
                        pdf_hash,
                        page_num,
                        required_version=image_processor.image_extraction_version,
                    )
                ]
                if pages_needing_images:
                    logger.info(
                        f"[{filename}] Extracting images from {len(pages_needing_images)} pages")
                    image_docs = image_processor.extract_images_from_pdf(
                        pdf_path=pdf_file,
                        pdf_hash=pdf_hash,
                        pdf_filename=filename,
                        ocr_text_per_page=ocr_text_per_page,
                    )
                    if image_docs:
                        image_vdb.add_documents(image_docs)

            if not text_done:
                mark_file_as_processed(filename)
            if not image_done:
                mark_image_as_processed(filename)
            logger.info(f"Completed: {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info("ETL (FULL) pipeline completed!")


def run_flask_api(port=5000):
    """Launch Flask API server."""
    logger.info(f"Starting Flask API server on port {port}...")
    from src.app.api import run_api
    run_api(port=port)


def run_export_image_review(output_path: str, pdf_filename: str = "", include_completed: bool = False):
    """Export extracted image metadata for human review/editing."""
    from src.etl import ImageReviewManager

    manager = ImageReviewManager()
    count = manager.export_for_review(
        output_path=output_path,
        pdf_filename=pdf_filename.strip() or None,
        only_pending=not include_completed,
    )
    logger.info(f"Exported {count} image review rows to {output_path}")


def run_export_image_db(output_path: str, pdf_filename: str = ""):
    """Export full image metadata DB snapshot from manifest records."""
    from src.etl import ImageReviewManager

    manager = ImageReviewManager()
    count = manager.export_db_snapshot(
        output_path=output_path,
        pdf_filename=pdf_filename.strip() or None,
    )
    logger.info(f"Exported {count} image DB rows to {output_path}")


def run_upsert_image_review_item(item_path: str, reviewed_by: str = "human"):
    """Upsert a single image metadata JSON object into review DB + vector DB."""
    from src.etl import ImageReviewManager

    payload = json.loads(Path(item_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--upsert-image-review-item expects a JSON object file")

    manager = ImageReviewManager()
    result = manager.upsert_review_item(item=payload, reviewed_by=reviewed_by)
    logger.info(f"Upserted image review item: {result}")


def run_apply_image_review(review_path: str, reviewed_by: str = "human", pdf_filename: str = ""):
    """Apply human-edited image metadata and sync image vector DB."""
    from src.etl import ImageReviewManager

    manager = ImageReviewManager()
    result = manager.apply_review_updates(
        review_path=review_path,
        reviewed_by=reviewed_by,
        pdf_filename=pdf_filename.strip() or None,
    )
    logger.info(f"Applied image review updates: {result}")


def run_replace_image_db(snapshot_path: str, reviewed_by: str = "human"):
    """Replace image metadata manifest and image vector DB from a snapshot JSON array."""
    from src.etl import ImageReviewManager

    manager = ImageReviewManager()
    result = manager.replace_image_db(snapshot_path=snapshot_path, reviewed_by=reviewed_by)
    logger.info(f"Replaced image DB from snapshot: {result}")


def run_import_images_dir(directory: str):
    """Import a local directory of images, bypassing PDF extraction."""
    from src.etl.local_image_importer import LocalImageImporter
    from src.rag.image_vectorstore import ImageVectorDB

    logger.info(f"Starting local image import from {directory}...")
    importer = LocalImageImporter()
    image_docs = importer.import_directory(directory)
    
    if image_docs:
        logger.info(f"Adding {len(image_docs)} images to ImageVectorDB...")
        image_vdb = ImageVectorDB()
        image_vdb.add_documents(image_docs)
        logger.info("Import completed successfully.")
    else:
        logger.warning("No images imported.")


def main():
    parser = argparse.ArgumentParser(description="Biology RAG System")

    etl_group = parser.add_argument_group("ETL Options")
    etl_group.add_argument("--etl", action="store_true",
                           help="Run full ETL pipeline (text + images)")
    etl_group.add_argument(
        "--text-only",
        action="store_true",
        help="Run ETL for text only. Skips pages already indexed.",
    )
    etl_group.add_argument(
        "--image-only",
        action="store_true",
        help="Run ETL for images only. Skips pages already processed.",
    )
    etl_group.add_argument(
        "--export-image-review",
        type=str,
        default="",
        help="Export image metadata review file (JSON) to this path.",
    )
    etl_group.add_argument(
        "--export-image-db",
        type=str,
        default="",
        help="Export full image metadata DB snapshot (JSON) to this path.",
    )
    etl_group.add_argument(
        "--apply-image-review",
        type=str,
        default="",
        help="Apply reviewed image metadata from a JSON file.",
    )
    etl_group.add_argument(
        "--replace-image-db",
        type=str,
        default="",
        help="Replace image metadata DB from a JSON snapshot and rebuild image index.",
    )
    etl_group.add_argument(
        "--upsert-image-review-item",
        type=str,
        default="",
        help="Upsert one image review item from a JSON object file.",
    )
    etl_group.add_argument(
        "--import-images-dir",
        type=str,
        default="",
        help="Import images directly from a local directory, bypassing PDF extraction.",
    )
    etl_group.add_argument(
        "--review-pdf",
        type=str,
        default="",
        help="Optional PDF filename filter for --export-image-review, --export-image-db, and --apply-image-review.",
    )
    etl_group.add_argument(
        "--review-user",
        type=str,
        default="human",
        help="Reviewer name used by review/apply/upsert/replace commands.",
    )
    etl_group.add_argument(
        "--review-include-completed",
        action="store_true",
        help="Include approved/rejected rows when exporting review file.",
    )
    etl_group.add_argument(
        "--ocr-strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "paddle", "tesseract", "pdf_text"],
        help="OCR strategy: hybrid (PDF text->PaddleOCR+VietOCR->Tesseract), "
             "paddle (PaddleOCR+VietOCR only), tesseract (enhanced Tesseract only), "
             "pdf_text (PDF text layer only). Default: hybrid",
    )

    parser.add_argument("--api", action="store_true",
                        help="Launch Flask API server")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for Flask API server (default: 5000)")

    args = parser.parse_args()

    if (
        not args.etl
        and not args.text_only
        and not args.image_only
        and not args.api
        and not args.export_image_review
        and not args.export_image_db
        and not args.apply_image_review
        and not args.replace_image_db
        and not args.upsert_image_review_item
        and not args.import_images_dir
    ):
        parser.print_help()
        sys.exit(1)

    if args.import_images_dir:
        run_import_images_dir(args.import_images_dir)

    if args.text_only:
        run_etl_text_only(ocr_strategy=args.ocr_strategy)
    elif args.image_only:
        run_etl_image_only()
    elif args.etl:
        run_etl(ocr_strategy=args.ocr_strategy)

    if args.export_image_review:
        run_export_image_review(
            output_path=args.export_image_review,
            pdf_filename=args.review_pdf,
            include_completed=args.review_include_completed,
        )

    if args.export_image_db:
        run_export_image_db(
            output_path=args.export_image_db,
            pdf_filename=args.review_pdf,
        )

    if args.apply_image_review:
        run_apply_image_review(
            review_path=args.apply_image_review,
            reviewed_by=args.review_user,
            pdf_filename=args.review_pdf,
        )

    if args.replace_image_db:
        run_replace_image_db(snapshot_path=args.replace_image_db, reviewed_by=args.review_user)

    if args.upsert_image_review_item:
        run_upsert_image_review_item(item_path=args.upsert_image_review_item, reviewed_by=args.review_user)

    if args.api:
        run_flask_api(port=args.port)


if __name__ == "__main__":
    main()
