"""Custom Gemini LLM client with retry and exponential backoff."""
import os
import asyncio
import httpx


class GeminiLLMClient:
    """Gemini API client with automatic retry on rate limits."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        contents = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": max_tokens if max_tokens is not None else self.max_tokens,
                "topP": 0.95,
            }
        }

        headers = {"Content-Type": "application/json"}

        # Retry with exponential backoff
        for attempt in range(8):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    feedback = data.get("promptFeedback", {})
                    block_reason = feedback.get("blockReason", "unknown")
                    raise RuntimeError(f"Gemini blocked: {block_reason}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = (2 ** attempt) * 5 + 10
                    print(f"[Gemini] Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise
            except httpx.TimeoutException:
                wait = (2 ** attempt) * 3
                print(f"[Gemini] Timeout, retrying in {wait}s...")
                await asyncio.sleep(wait)

        raise RuntimeError("Gemini: max retries exceeded")
