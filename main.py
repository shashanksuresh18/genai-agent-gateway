import os
import uuid
import json
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import HTTPError, URLError

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="genai-agent-gateway", version="0.2.0")

# -----------------------------
# Feature flags / gateway auth
# -----------------------------
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()

# -----------------------------
# Azure OpenAI config (from env)
# -----------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

# Keep output small for demos/tests
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "256"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Keep answers concise and safe.",
).strip()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


def _require_gateway_key(x_api_key: str | None):
    # If you set GATEWAY_API_KEY, then enforce it.
    if GATEWAY_API_KEY and x_api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid x-api-key")


def _azure_openai_chat(user_message: str, request_id: str) -> str:
    # Basic config checks
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        raise HTTPException(
            status_code=500,
            detail="Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT.",
        )

    # Azure OpenAI Chat Completions endpoint
    base = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
    url = f"{base}?{urlencode({'api-version': AZURE_OPENAI_API_VERSION})}"

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # IMPORTANT for your model: use max_completion_tokens (not max_tokens)
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        # DO NOT send temperature — your model rejects non-default values.
        # If you send it, Azure will throw "Unsupported value".
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
        "x-ms-client-request-id": request_id,
    }

    data = json.dumps(payload).encode("utf-8")
    req = UrlRequest(url, data=data, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            out = json.loads(body)
    except HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else str(e)
        raise HTTPException(
            status_code=502,
            detail=f"Azure OpenAI HTTP {e.code}: {err_body}",
        )
    except URLError as e:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI connection error: {e.reason}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI unexpected error: {repr(e)}")

    # Parse response
    try:
        return out["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI returned unexpected payload: {out}")


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


@app.post("/api/chat")
def chat(req: ChatRequest, x_api_key: str | None = Header(default=None, alias="x-api-key")):
    _require_gateway_key(x_api_key)

    request_id = str(uuid.uuid4())

    # simple safety filter
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

    answer = _azure_openai_chat(req.message, request_id)
    return {"request_id": request_id, "answer": answer, "mock_llm": False}
