"""LLM initialization using HuggingFace models or Gemini API."""

import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS, LLM_TOP_P, HF_TOKEN, GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


def get_hf_llm(
    model_name: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    top_p: float = LLM_TOP_P,
    **kwargs
):
    """Initialize HuggingFace LLM with Qwen model."""
    logger.info(f"Loading LLM: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN if HF_TOKEN else None)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_length=None,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=top_p,
    )

    llm = HuggingFacePipeline(pipeline=model_pipeline, model_kwargs=kwargs)
    logger.info("LLM initialized successfully")
    return llm


def get_gemini_llm(
    model: str = GEMINI_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_NEW_TOKENS,
    api_key: str = GEMINI_API_KEY,
):
    """Initialize Gemini API LLM."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to use Gemini LLM")
    logger.info(f"Loading Gemini LLM: {model}")
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        transport="rest",
    )
    logger.info("Gemini LLM initialized successfully")
    return llm


def get_llm():
    """Get LLM — prefers Gemini if API key is set, otherwise HuggingFace."""
    if GEMINI_API_KEY:
        return get_gemini_llm()
    return get_hf_llm()
