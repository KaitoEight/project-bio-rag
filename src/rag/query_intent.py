"""Query intent helpers shared by text and image retrieval."""

import re
import unicodedata


IMAGE_QUERY_HINTS = {
    "anh minh hoa",
    "anh ve",
    "cho minh anh",
    "cho minh hinh",
    "cho toi anh",
    "cho toi hinh",
    "cho xem anh",
    "cho xem hinh",
    "hinh anh",
    "hinh minh hoa",
    "hinh ve",
    "lay anh",
    "lay hinh",
    "minh hoa",
    "quan sat",
    "so do",
    "tim anh",
    "tim hinh",
    "tranh",
    "xem anh",
    "xem hinh",
}

IMAGE_ONLY_HINTS = {
    "anh thoi",
    "chi can anh",
    "chi can hinh",
    "chi xem anh",
    "chi xem hinh",
    "hinh thoi",
    "image only",
    "only image",
}

IMAGE_QUERY_ACTIONS = {
    "can",
    "cho",
    "co",
    "dua",
    "gui",
    "hay",
    "lay",
    "muon",
    "show",
    "tim",
    "toi",
    "xem",
}
IMAGE_QUERY_FILLERS = {"minh", "toi", "tui", "em", "anh", "chi", "ve", "cua"}
IMAGE_QUERY_NOUNS = {"anh", "hinh", "photo", "picture", "tranh"}

TEXT_ANSWER_HINTS = {
    "cho biet",
    "dinh nghia",
    "giai thich",
    "khai niem",
    "la gi",
    "mo ta",
    "noi dung",
    "phan tich",
    "the nao",
    "tom tat",
    "tra loi",
    "trinh bay",
    "vi sao",
    "what is",
    "why",
}


def normalize_query_text(text: str) -> str:
    """Normalize Vietnamese text to simple accentless tokens."""
    normalized = unicodedata.normalize("NFD", str(text or "").lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", normalized)).strip()


def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics from already-normalized text."""
    decomposed = unicodedata.normalize("NFD", str(text or ""))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_accented_text(text: str) -> str:
    """Lowercase and keep letters/digits + single spaces, PRESERVING diacritics.

    Unlike :func:`normalize_query_text`, this keeps Vietnamese tone/vowel marks so
    that distinct words are not conflated — e.g. "trâu" (buffalo) stays different
    from "trầu" (betel). Use this for exact lexical/phrase matching; accent folding
    is only safe for the accent-free intent-keyword lookups.
    """
    lowered = unicodedata.normalize("NFC", str(text or "").lower())
    kept = [char if (char.isalnum() or char.isspace()) else " " for char in lowered]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def has_image_intent(query: str) -> bool:
    normalized_query = normalize_query_text(query)
    if not normalized_query:
        return False
    if any(term in normalized_query for term in IMAGE_QUERY_HINTS):
        return True

    tokens = normalized_query.split()
    for index, token in enumerate(tokens):
        if token not in IMAGE_QUERY_NOUNS:
            continue
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token == "anh" and next_token == "sang":
            continue
        if token == "anh" and next_token in {"biet", "co", "giup", "la", "noi"}:
            continue
        if token == "hinh" and next_token in {"thanh", "thuc"}:
            continue
        if index == 0 or index <= 3:
            return True
        previous_tokens = set(tokens[max(0, index - 4) : index])
        if previous_tokens & (IMAGE_QUERY_ACTIONS | IMAGE_QUERY_FILLERS):
            return True
    return False


def is_image_only_query(query: str) -> bool:
    """Return True when the user appears to request only image results."""
    normalized_query = normalize_query_text(query)
    if not has_image_intent(normalized_query):
        return False
    if any(term in normalized_query for term in IMAGE_ONLY_HINTS):
        return True
    if any(term in normalized_query for term in TEXT_ANSWER_HINTS):
        return False

    tokens = normalized_query.split()
    if not tokens:
        return False

    first_image_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token in IMAGE_QUERY_NOUNS
            and not (token == "anh" and index + 1 < len(tokens) and tokens[index + 1] == "sang")
            and not (
                token == "anh"
                and index + 1 < len(tokens)
                and tokens[index + 1] in {"biet", "co", "giup", "la", "noi"}
            )
            and not (token == "hinh" and index + 1 < len(tokens) and tokens[index + 1] in {"thanh", "thuc"})
        ),
        None,
    )
    if first_image_index is None:
        return False

    prefix_tokens = set(tokens[:first_image_index])
    return first_image_index <= 4 and prefix_tokens <= (IMAGE_QUERY_ACTIONS | IMAGE_QUERY_FILLERS)
