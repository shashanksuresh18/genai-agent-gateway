import os
import uuid
import secrets

from fastapi import FastAPI, Request, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="genai-agent-gateway")

MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"

# If API_KEY is set (recommended in Azure), /api/chat requires header: x-api-key: <value>
API_KEY = os.getenv("API_KEY")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def require_api_key(x_api_key: str | None = Header(default=None, alias="x-api-key")):
    """
    Minimal auth gate:
    - If API_KEY is NOT set -> allow (dev convenience)
    - If API_KEY is set -> require matching x-api-key header
    """
    if not API_KEY:
        return
    if (not x_api_key) or (not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.get("/")
def home():
    return FileResponse("web/index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request, _: None = Depends(require_api_key)):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # This is NOT real security. It's just a small guardrail to avoid responding to obvious credential exfiltration prompts.
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
        # We’ll wire Azure OpenAI later. For now keep the shape stable.
        answer = "LLM mode not wired yet. Set MOCK_LLM=true."

    return {"request_id": request_id, "answer": answer, "mock_llm": MOCK_LLM}
