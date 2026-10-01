"""OpenAI Realtime <-> Home Assistant bridge.

- Serves the browser UI (via HA ingress).
- Mints short-lived Realtime client secrets (the real API key never reaches the browser).
- Executes tool calls the model makes against the ha-mcp server, with an optional
  confirmation gate for destructive operations.
"""
import json
import logging
import time
import re
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from mcp import ClientSession
import contextlib
try:
    from mcp.client.streamable_http import streamable_http_client as _client
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client as _client


@contextlib.asynccontextmanager
async def streamablehttp_client(url):
    # New SDK yields (read, write); old SDK yields (read, write, get_session_id).
    async with _client(url) as streams:
        yield streams[0], streams[1], None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("realtime-voice")

OPTIONS = json.loads(Path("/data/options.json").read_text())
API_KEY = OPTIONS.get("openai_api_key", "")
MCP_URL = OPTIONS.get("ha_mcp_url", "")
MODEL = OPTIONS.get("model", "gpt-realtime-2.1")
ALLOW = {t.strip() for t in str(OPTIONS.get("tool_allowlist", "")).split(",") if t.strip()}
VOICE = OPTIONS.get("voice", "marin")
CONFIRM = bool(OPTIONS.get("confirm_destructive", True))
MAX_OUT = int(OPTIONS.get("max_tool_output_chars", 12000))
INSTRUCTIONS = OPTIONS.get("instructions", "")

# Home Assistant ingress connects from the supervisor gateway only.
ALLOWED_CLIENTS = {"172.30.32.2", "127.0.0.1", "::1"}
DESTRUCTIVE = re.compile(r"(restart|remove|delete|reload_core)", re.I)

app = FastAPI()
STATIC = Path(__file__).parent / "static"


@app.middleware("http")
async def ingress_only(request: Request, call_next):
    client = request.client.host if request.client else ""
    if client not in ALLOWED_CLIENTS:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return await call_next(request)


async def list_tools() -> list[dict]:
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    tools = []
    all_names = [t.name for t in result.tools]
    for t in result.tools:
        if ALLOW and t.name not in ALLOW:
            continue
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        schema.setdefault("type", "object")
        tools.append(
            {
                "type": "function",
                "name": t.name,
                "description": (t.description or "")[:800],
                "parameters": schema,
            }
        )
    if ALLOW:
        missing = sorted(ALLOW - set(all_names))
        if missing:
            log.warning("allowlisted tools not found on server: %s", missing)
    log.info("tools: %d of %d exposed, schema %d bytes", len(tools), len(all_names), len(json.dumps(tools)))
    return tools


def is_destructive(name: str, args: dict) -> bool:
    if DESTRUCTIVE.search(name):
        return True
    for key, value in (args or {}).items():
        if isinstance(value, str) and DESTRUCTIVE.search(value) and key in {
            "tool", "tool_name", "name", "action", "operation",
        }:
            return True
    return False


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.post("/session")
async def create_session():
    if not API_KEY:
        return JSONResponse({"error": "Set openai_api_key in the add-on options."}, 400)
    if not MCP_URL:
        return JSONResponse({"error": "Set ha_mcp_url in the add-on options."}, 400)
    try:
        tools = await list_tools()
    except Exception as exc:  # noqa: BLE001
        log.exception("tool listing failed")
        return JSONResponse({"error": f"Could not reach ha-mcp: {exc}"}, 502)

    body = {
        "session": {
            "type": "realtime",
            "model": MODEL,
            "instructions": INSTRUCTIONS,
            "tools": tools,
            "tool_choice": "auto",
            "audio": {
                "input": {
                    "transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {"type": "semantic_vad"},
                },
                "output": {"voice": VOICE},
            },
        }
    }
    t0 = time.time()
    log.info("OpenAI client_secrets -> model=%s tools=%d payload=%d bytes", MODEL, len(tools), len(json.dumps(body)))
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://api.openai.com/v1/realtime/client_secrets",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=body,
        )
    log.info("OpenAI client_secrets <- %s in %d ms", resp.status_code, (time.time() - t0) * 1000)
    if resp.status_code >= 400:
        log.error("client_secrets failed: %s %s", resp.status_code, resp.text[:300])
        return JSONResponse(
            {"error": f"OpenAI error {resp.status_code}: {resp.text[:300]}"}, 502
        )
    data = resp.json()
    return {"client_secret": data.get("value"), "tool_count": len(tools), "model": MODEL}


@app.post("/clientlog")
async def clientlog(request: Request):
    try:
        d = await request.json()
    except Exception:  # noqa: BLE001
        return {"ok": False}
    log.info("browser: %s", str(d.get("msg", ""))[:500])
    return {"ok": True}


@app.post("/tool")
async def call_tool(request: Request):
    payload = await request.json()
    name = payload.get("name", "")
    args = payload.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return JSONResponse({"error": "bad arguments"}, 400)
    confirmed = bool(payload.get("confirmed"))

    if CONFIRM and is_destructive(name, args) and not confirmed:
        return {
            "needs_confirmation": True,
            "summary": f"{name}({json.dumps(args)[:300]})",
        }

    log.info("tool call: %s args=%s", name, json.dumps(args)[:300])
    tc0 = time.time()
    try:
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
    except Exception as exc:  # noqa: BLE001
        log.exception("tool call failed")
        return {"output": json.dumps({"error": str(exc)})}

    log.info("tool done: %s in %d ms", name, (time.time() - tc0) * 1000)
    text = "\n".join(
        c.text for c in result.content if getattr(c, "type", "") == "text"
    )
    if len(text) > MAX_OUT:
        text = text[:MAX_OUT] + "\n...[truncated]"
    if getattr(result, "is_error", getattr(result, "isError", False)):
        text = json.dumps({"error": text})
    return {"output": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8099, log_level="info")
