# genai-agent-gateway

Minimal FastAPI + tiny web UI.

Endpoints:
- GET /healthz
- POST /api/chat  {"message": "..."}
- GET / (web UI)

Environment:
- MOCK_LLM=true (default)
