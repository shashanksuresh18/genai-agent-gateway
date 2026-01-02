import os
import uuid
import time
import hmac
import json
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="genai-agent-gateway", version="0.2.0")

# -----------------------------
# Config (env vars)
# -----------------------------
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

# Auth
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "").strip()  # for programmatic clients
GATEWAY_UI_PASSWORD = os.getenv("GATEWAY_UI_PASSWORD", "").strip()  # for browser login
SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()  # signing secret for cookie (required if UI password enabled)
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "43200"))  # 12h default

# Cookie name
SESSION_COOKIE = "gateway_session"

WEB_DIR = Path(__file__).parent / "web"
INDEX_HTML = WEB_DIR / "index.html"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=200)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


# -----------------------------
# Session cookie (stateless signed)
# -----------------------------
def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(secret: str, payload_bytes: bytes) -> bytes:
    return hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).digest()


def create_session_token(secret: str, ttl_seconds: int) -> str:
    payload = {
        "v": 1,
        "exp": int(time.time()) + ttl_seconds,
        "iat": int(time.time()),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = _sign(secret, payload_bytes)
    return f"{_b64url_encode(payload_bytes)}.{_b64url_encode(sig)}"


def verify_session_token(secret: str, token: str) -> bool:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)

        expected = _sign(secret, payload_bytes)
        if not hmac.compare_digest(sig, expected):
            return False

        payload = json.loads(payload_bytes.decode("utf-8"))
        exp = int(payload.get("exp", 0))
        return time.time() < exp
    except Exception:
        return False


def is_browser_session_authenticated(request: Request) -> bool:
    if not (GATEWAY_UI_PASSWORD and SESSION_SECRET):
        return False
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return False
    return verify_session_token(SESSION_SECRET, token)


def require_auth(request: Request) -> None:
    """
    Accept either:
      - valid browser session cookie (UI)
      - valid x-api-key header (API clients)
    """
    if is_browser_session_authenticated(request):
        return

    if GATEWAY_API_KEY:
        key = request.headers.get("x-api-key", "")
        if key and hmac.compare_digest(key, GATEWAY_API_KEY):
            return
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid x-api-key.")

    # If no API key configured, fall back to UI password/session requirement if configured.
    if GATEWAY_UI_PASSWORD and SESSION_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized: please login.")

    # If neither auth configured, allow (not recommended)
    return


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="web/index.html not found in container.")
    return FileResponse(str(INDEX_HTML))


@app.get("/healthz")
def healthz(request: Request):
    return {
        "ok": True,
        "mock_llm": MOCK_LLM,
        "has_gateway_api_key": bool(GATEWAY_API_KEY),
        "has_ui_password": bool(GATEWAY_UI_PASSWORD),
        "has_session_secret": bool(SESSION_SECRET),
        "has_azure_openai_endpoint": bool(AZURE_OPENAI_ENDPOINT),
        "has_azure_openai_key": bool(AZURE_OPENAI_API_KEY),
        "has_azure_openai_deployment": bool(AZURE_OPENAI_DEPLOYMENT),
        "azure_openai_api_version": AZURE_OPENAI_API_VERSION,
        "ui_session_authenticated": is_browser_session_authenticated(request),
    }


@app.post("/auth/login")
def login(req: LoginRequest, response: Response, request: Request):
    if not GATEWAY_UI_PASSWORD:
        raise HTTPException(status_code=501, detail="UI login is not enabled (GATEWAY_UI_PASSWORD missing).")
    if not SESSION_SECRET:
        raise HTTPException(status_code=500, detail="SESSION_SECRET missing. Set it to enable signed sessions.")

    if not hmac.compare_digest(req.password, GATEWAY_UI_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid password.")

    token = create_session_token(SESSION_SECRET, SESSION_TTL_SECONDS)
    secure_cookie = request.url.scheme == "https"
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
        path="/",
    )
    return {"ok": True}


@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(key=SESSION_COOKIE, path="/")
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    require_auth(request)

    request_id = str(uuid.uuid4())
    lower = req.message.lower()

    # Safety: refuse obvious secret/bypass attempts
    if any(x in lower for x in ["api key", "client secret", "password", "token", "system prompt", "bypass"]):
        return {
            "request_id": request_id,
            "answer": "I can’t help with credentials or bypassing controls. Describe the goal and I’ll suggest a safe approach.",
            "mock_llm": MOCK_LLM,
        }

    if MOCK_LLM:
        answer = f"[MOCK] Received: {req.message[:300]}"
        return {"request_id": request_id, "answer": answer, "mock_llm": True}

    # Validate Azure OpenAI config
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT):
        raise HTTPException(
            status_code=500,
            detail="Azure OpenAI config missing. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT.",
        )

    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    # IMPORTANT:
    # Your model previously rejected max_tokens / some temperature values.
    # So we keep the payload minimal and let the service defaults apply.
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep answers concise."},
            {"role": "user", "content": req.message},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
        "x-request-id": request.headers.get("x-request-id", request_id),
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Azure OpenAI HTTP {r.status_code}: {r.text}")
            data = r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI request failed: {type(e).__name__}: {str(e)}")

    # Extract assistant message
    answer: Optional[str] = None
    try:
        answer = data["choices"][0]["message"]["content"]
    except Exception:
        answer = json.dumps(data)[:1000]

    return {"request_id": request_id, "answer": answer, "mock_llm": False}
