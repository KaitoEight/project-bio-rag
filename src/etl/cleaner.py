"""Text cleaning utilities for Vietnamese content."""

import re
import unicodedata


def clean_vietnamese_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("C") or char in "\n\t"
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def clean_gemini_ocr_text(text: str) -> str:
    """Clean text produced by Gemini Vision OCR.

    Handles common Gemini output artifacts:
    - Markdown formatting (bold, headers, code blocks)
    - Leading/trailing boilerplate
    - Excessive whitespace
    """
    if not text:
        return ""

    # Normalize unicode first
    text = unicodedata.normalize("NFC", text)

    # Remove markdown code block wrappers
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Remove markdown bold/italic markers but keep the text
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)

    # Remove markdown headers (keep the text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove invisible/control characters (keep newlines and tabs)
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("C") or char in "\n\t"
    )

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()

