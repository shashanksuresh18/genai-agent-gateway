import os
import uuid
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Azure OpenAI SDK (OpenAI Python client)
from openai import AzureOpenAI

app = FastAPI(title="genai-agent-gateway")

# Toggles
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"

# Optional gateway API key (if set, requests must include x-api-key header)
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()

# Azure OpenAI config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()  # e.g. https://aoai-gateway-shanky-01.openai.azure.com
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


def _enforce_gateway_key(x_api_key: Optional[str]):
    if not GATEWAY_API_KEY:
        return
    if not x_api_key or x_api_key.strip() != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


def _build_client() -> AzureOpenAI:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("Azure OpenAI endpoint/key missing. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


@app.post("/api/chat")
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    request_id = str(uuid.uuid4())

    # Optional gateway auth
    _enforce_gateway_key(x_api_key)

    # Very basic “bank-ish” safety: refuse obvious credential/bypass asks
    lower = req.message.lower()
    if any(x in lower for x in ["api key", "client secret", "password", "token", "system prompt", "bypass"]):
        return {
            "request_id": request_id,
            "answer": "I can’t help with credentials or bypassing controls. Describe the goal and I’ll suggest a safe approach.",
            "mock_llm": MOCK_LLM,
        }

    if MOCK_LLM:
        answer = f"[MOCK] Received: {req.message[:300]}"
        return {"request_id": request_id, "answer": answer, "mock_llm": True}

    # Real Azure OpenAI call
    if not AZURE_OPENAI_DEPLOYMENT:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_DEPLOYMENT is not set (e.g., gateway-chat).")

    try:
        client = _build_client()
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,  # IMPORTANT: this is the DEPLOYMENT NAME, not the base model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": req.message},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return {"request_id": request_id, "answer": answer, "mock_llm": False}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI call failed: {type(e).__name__}: {e}")
