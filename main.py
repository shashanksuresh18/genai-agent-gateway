import os
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="genai-agent-gateway")

# ===== Config =====
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"

# Gateway auth (your own API key to protect /api/chat)
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()

# Azure OpenAI config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()  # e.g. https://aoai-xxx.openai.azure.com
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()  # e.g. gateway-chat


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.get("/")
def home():
    return FileResponse("web/index.html")


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "mock_llm": MOCK_LLM,
        "has_gateway_api_key": bool(GATEWAY_API_KEY),
        "has_azure_openai_endpoint": bool(AZURE_OPENAI_ENDPOINT),
        "has_azure_openai_key": bool(AZURE_OPENAI_API_KEY),
        "has_azure_openai_deployment": bool(AZURE_OPENAI_DEPLOYMENT),
        "azure_openai_api_version": AZURE_OPENAI_API_VERSION,
    }


def _require_gateway_key(x_api_key: Optional[str]) -> None:
    # If you asked for auth, we enforce it. If key not configured, fail loudly.
    if not GATEWAY_API_KEY:
        raise HTTPException(status_code=500, detail="GATEWAY_API_KEY is not configured on the server.")
    if not x_api_key or x_api_key.strip() != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid x-api-key.")


def _call_azure_openai(user_text: str) -> str:
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT):
        raise RuntimeError("Azure OpenAI is not configured. Set endpoint, key, and deployment.")

    url = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}"
        f"/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    # IMPORTANT:
    # - Don't set temperature (your model complained when non-default was used)
    # - Some models want max_completion_tokens instead of max_tokens, so we try both.
    base_payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text[:4000]},
        ]
    }

    # Try newer param first, then fallback for compatibility.
    for token_param in ("max_completion_tokens", "max_tokens"):
        payload = dict(base_payload)
        payload[token_param] = 256

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                return str(data)

        # If it's the "unsupported parameter" error, try the other param
        if resp.status_code == 400 and "unsupported" in resp.text.lower():
            continue

        # Anything else: raise
        raise RuntimeError(f"Azure OpenAI HTTP {resp.status_code}: {resp.text}")

    raise RuntimeError("Azure OpenAI call failed for both token parameter styles.")


@app.post("/api/chat")
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(default=None)):
    # Enforce gateway auth
    _require_gateway_key(x_api_key)

    request_id = str(uuid.uuid4())

    # Simple “bank-ish” safety: refuse obvious secret/bypass attempts
    lower = req.message.lower()
    if any(x in lower for x in ["api key", "client secret", "password", "token", "system prompt", "bypass"]):
        return {
            "request_id": request_id,
            "answer": "I can’t help with credentials or bypassing controls. Describe the goal and I’ll suggest a safe approach.",
            "mock_llm": MOCK_LLM,
        }

    if MOCK_LLM:
        answer = f"[MOCK] Received: {req.message[:300]}"
    else:
        try:
            answer = _call_azure_openai(req.message)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    return {"request_id": request_id, "answer": answer, "mock_llm": MOCK_LLM}
