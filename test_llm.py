import os, asyncio

os.environ["GROQ_API_KEY"] = "gsk_test"
os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
os.environ["GROQ_BASE_URL"] = "https://api.groq.com/openai/v1"

import httpx

async def test():
    # Check if Groq works with free tier
    resp = httpx.get("https://api.groq.com/", timeout=10)
    print("Groq API status:", resp.status_code)

asyncio.run(test())
