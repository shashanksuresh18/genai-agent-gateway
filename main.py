import os
import json
import uuid
import urllib.request
import urllib.error
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="genai-agent-gateway")

# --- Feature flags / gateway auth ---
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()  # optional

# --- Azure OpenAI config (from Container App env vars) ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

# Optional tuning
AOAI_TEMPERATURE = float(os.getenv("AOAI_TEMPERATURE", "0.2"))
AOAI_MAX_OUTPUT_TOKENS = int(os.getenv("AOAI_MAX_OUTPUT_TOKENS", "300"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Be concise and safe. Do not reveal secrets.",
).strip()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


def _require_gateway_key(x_api_key: Optional[str]):
    # Only enforce if you actually configured a gateway key
    if not GATEWAY_API_KEY:
        return
    if not x_api_key or x_api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


def _aoai_url() -> str:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
        return ""
    return (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )


def _call_aoai(messages, request_id: str) -> str:
    """
    Calls Azure OpenAI Chat Completions via REST.
    Handles the 'max_tokens' vs 'max_completion_tokens' mismatch by retrying.
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        raise RuntimeError("Azure OpenAI is not configured (missing endpoint/key/deployment).")

    url = _aoai_url()

    base_payload = {
        "messages": messages,
        "temperature": AOAI_TEMPERATURE,
    }

    # Try modern parameter first (your model demanded this)
    payload_1 = dict(base_payload)
    payload_1["max_completion_tokens"] = AOAI_MAX_OUTPUT_TOKENS

    # Fallback for older models/APIs
    payload_2 = dict(base_payload)
    payload_2["max_tokens"] = AOAI_MAX_OUTPUT_TOKENS

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
        "x-ms-client-request-id": request_id,
    }

    def do_request(payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)

    try:
        resp_json = do_request(payload_1)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        # If server complains about max_completion_tokens, retry with max_tokens
        if "max_completion_tokens" in body and "Unrecognized" in body:
            resp_json = do_request(payload_2)
        # If server complains about max_tokens, retry with max_completion_tokens
        elif "max_tokens" in body and "Unsupported parameter" in body:
            resp_json = do_request(payload_1)
        else:
            raise RuntimeError(f"Azure OpenAI HTTP {e.code}: {body}") from None
    except Exception as e:
        raise RuntimeError(f"Azure OpenAI call failed: {repr(e)}") from None

    # Standard Chat Completions shape
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected Azure OpenAI response shape: {resp_json}") from None


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
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    _require_gateway_key(x_api_key)

    request_id = str(uuid.uuid4())

    # Simple safety: refuse obvious secret/bypass attempts
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.message},
    ]

    try:
        answer = _call_aoai(messages, request_id=request_id)
        return {"request_id": request_id, "answer": answer, "mock_llm": False}
    except Exception as e:
        # Keep it visible for debugging in Swagger, but don't dump secrets
        raise HTTPException(status_code=502, detail=str(e))
