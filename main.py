import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="genai-agent-gateway")

MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"


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
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest):
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
        # We’ll wire Azure OpenAI later. For now keep the shape stable.
        answer = "LLM mode not wired yet. Set MOCK_LLM=true."

    return {"request_id": request_id, "answer": answer, "mock_llm": MOCK_LLM}
