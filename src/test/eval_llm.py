"""LLM dùng cho sinh test + chấm (generator/judge), cấu hình qua biến môi trường.

Dùng giao thức OpenAI-compatible nên cắm được BẤT KỲ provider nào theo chuẩn này:
Xiaomi MiMo, Groq, OpenRouter, Together, vLLM tự host... Chỉ cần đặt 3 biến trong
`.env`:

    EVAL_LLM_BASE_URL = <endpoint>      # vd MiMo / Groq
    EVAL_LLM_API_KEY  = <token của bạn>
    EVAL_LLM_MODEL    = <model id>

Ví dụ cấu hình:
    # Groq (free)
    EVAL_LLM_BASE_URL=https://api.groq.com/openai/v1
    EVAL_LLM_MODEL=llama-3.3-70b-versatile
    # Xiaomi MiMo (điền base_url + model id theo nhà cung cấp token của bạn)
    EVAL_LLM_BASE_URL=https://api.<...>/v1
    EVAL_LLM_MODEL=mimo-2.5-pro

Tùy chọn:
    EVAL_LLM_API_KEY_ENV = GROQ_API_KEY   # nếu key của bạn đã nằm ở biến tên khác
"""

import os
from langchain_openai import ChatOpenAI

_DEFAULT_KEY_CANDIDATES = (
    "EVAL_LLM_API_KEY", "MIMO_API_KEY", "GROQ_API_KEY",
    "OPENROUTER_API_KEY", "OPENAI_API_KEY",
)


def _resolve_api_key() -> str:
    # Cho phép trỏ tới biến key có sẵn qua EVAL_LLM_API_KEY_ENV.
    alias = os.getenv("EVAL_LLM_API_KEY_ENV", "").strip()
    if alias and os.getenv(alias):
        return os.getenv(alias)
    for name in _DEFAULT_KEY_CANDIDATES:
        val = os.getenv(name, "").strip()
        if val:
            return val
    return ""


def is_configured() -> bool:
    return bool(os.getenv("EVAL_LLM_MODEL", "").strip() and _resolve_api_key())


def config_help() -> str:
    return (
        "Chưa cấu hình LLM đánh giá. Thêm vào .env:\n"
        "  EVAL_LLM_BASE_URL=<endpoint OpenAI-compatible>\n"
        "  EVAL_LLM_API_KEY=<token của bạn>\n"
        "  EVAL_LLM_MODEL=<model id>\n"
        "Ví dụ Groq (free): BASE_URL=https://api.groq.com/openai/v1, MODEL=llama-3.3-70b-versatile"
    )


def get_eval_llm(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """Tạo client LLM đánh giá từ env. Báo lỗi rõ ràng nếu thiếu cấu hình."""
    model = os.getenv("EVAL_LLM_MODEL", "").strip()
    api_key = _resolve_api_key()
    base_url = os.getenv("EVAL_LLM_BASE_URL", "").strip() or None

    if not model or not api_key:
        raise RuntimeError(config_help())

    print(f"[eval_llm] model={model} base_url={base_url or 'default(openai)'}")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=60,
        max_retries=2,
        **kwargs,
    )
