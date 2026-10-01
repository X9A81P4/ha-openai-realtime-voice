"""OpenAI Realtime <-> Home Assistant bridge (v0.3).

- Serves the browser UI (via HA ingress).
- Mints short-lived Realtime client secrets (the real API key never reaches the browser).
- Executes the model's tool calls against ha-mcp, with an optional confirmation gate.
- Lets the model hand big jobs to a background Claude Code agent (headless `claude -p`
  limited to the ha-mcp tools, no shell).
- Persists a JSONL log (/data/voice-log.jsonl) of transcripts, tool calls, agent jobs and
  browser diagnostics; viewable at /logs.
"""
import asyncio
import contextlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamable_http_client as _client
except ImportError:  # older SDK
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
VOICE = OPTIONS.get("voice", "marin")
CONFIRM = bool(OPTIONS.get("confirm_destructive", True))
MAX_OUT = int(OPTIONS.get("max_tool_output_chars", 12000))
INSTRUCTIONS = OPTIONS.get("instructions", "")
ALLOW = {t.strip() for t in str(OPTIONS.get("tool_allowlist", "")).split(",") if t.strip()}

ANTHROPIC_KEY = OPTIONS.get("anthropic_api_key") or ""
CLAUDE_TOKEN = OPTIONS.get("claude_oauth_token") or ""
AGENT_MODEL = OPTIONS.get("agent_model", "sonnet")
AGENT_TURNS = int(OPTIONS.get("agent_max_turns", 40))
AGENT_TIMEOUT = int(OPTIONS.get("agent_timeout_s", 600))
CONFIRM_AGENT = bool(OPTIONS.get("confirm_agent", True))
AUTO_STOP = bool(OPTIONS.get("auto_stop_idle", True))
IDLE_S = int(OPTIONS.get("idle_timeout_s", 120))


def _prices(spec: str) -> dict:
    d = {"audio_in": 32.0, "audio_out": 64.0, "audio_cached": 0.4, "text_in": 4.0, "text_out": 16.0, "text_cached": 0.4}
    for part in str(spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                d[k.strip()] = float(v)
            except ValueError:
                pass
    return d


PRICES = _prices(OPTIONS.get("prices_per_million", ""))  # USD per 1M tokens (estimate)

# Home Assistant ingress connects from the supervisor gateway only.
ALLOWED_CLIENTS = {"172.30.32.2", "127.0.0.1", "::1"}
DESTRUCTIVE = re.compile(r"(restart|remove|delete|reload_core)", re.I)
WRITE_PREFIXES = (
    "ha_set_", "ha_config_set_", "ha_config_remove_", "ha_config_delete_", "ha_remove_",
    "ha_manage_", "ha_call_service", "ha_bulk_control", "ha_restart", "ha_reload_core",
    "ha_call_event", "ha_eval_template",
)

app = FastAPI()
STATIC = Path(__file__).parent / "static"
LOGFILE = Path("/data/voice-log.jsonl")
JOBS: dict[str, dict] = {}
TOOL_NAMES: list[str] = []
TOOL_META: list[dict] = []
TOOL_STATS: dict[str, dict] = {}

LOCAL_TOOLS = [
    {
        "type": "function",
        "name": "delegate_to_claude_agent",
        "description": (
            "Hand a larger or multi-step Home Assistant job to a Claude Code agent that works in "
            "the background (build or edit automations, scripts, dashboards, helpers; audit or "
            "clean up entities; investigate a problem). Returns a job id immediately, so tell the "
            "user it has started; the result is announced when it finishes, or use "
            "check_claude_agent. Do not use for quick lookups or simple on/off control."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Complete, self-contained instructions for the agent."},
                "allow_changes": {
                    "type": "boolean",
                    "description": "true lets the agent change Home Assistant; false (default) is read-only investigation.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "type": "function",
        "name": "check_claude_agent",
        "description": "Status and result of background Claude agent jobs. Omit job_id for the most recent job.",
        "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}},
    },
]


def flog(kind: str, **kw) -> None:
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **kw}
    try:
        if LOGFILE.exists() and LOGFILE.stat().st_size > 2_000_000:
            LOGFILE.replace(LOGFILE.with_suffix(".1.jsonl"))
        with LOGFILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


@app.middleware("http")
async def ingress_only(request: Request, call_next):
    client = request.client.host if request.client else ""
    if client not in ALLOWED_CLIENTS:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return await call_next(request)


async def list_tools(raw: bool = False) -> list[dict]:
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    all_names = [t.name for t in result.tools]
    TOOL_NAMES[:] = all_names
    TOOL_META[:] = []
    for t in result.tools:
        sc = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
        props = sc.get("properties", {}) or {}
        TOOL_META.append({
            "name": t.name, "description": (t.description or "").strip(),
            "params": [{"name": k, "type": (v.get("type") if isinstance(v, dict) else None),
                        "required": k in (sc.get("required") or []),
                        "description": ((v.get("description") or "")[:160] if isinstance(v, dict) else "")}
                       for k, v in props.items()],
            "schema_bytes": len(json.dumps(sc)),
        })
    tools = []
    for t in result.tools:
        if ALLOW and t.name not in ALLOW:
            continue
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {
            "type": "object", "properties": {}}
        schema.setdefault("type", "object")
        tools.append({"type": "function", "name": t.name,
                      "description": (t.description or "")[:800], "parameters": schema})
    if ALLOW:
        missing = sorted(ALLOW - set(all_names))
        if missing:
            log.warning("allowlisted tools not found on server: %s", missing)
    log.info("tools: %d of %d exposed, schema %d bytes", len(tools), len(all_names), len(json.dumps(tools)))
    return tools


def is_destructive(name: str, args: dict) -> bool:
    if name == "delegate_to_claude_agent":
        return CONFIRM_AGENT
    if DESTRUCTIVE.search(name):
        return True
    for key, value in (args or {}).items():
        if isinstance(value, str) and DESTRUCTIVE.search(value) and key in {
            "tool", "tool_name", "name", "action", "operation",
        }:
            return True
    return False


# ---------------------------------------------------------------- agent jobs
def agent_env() -> dict:
    env = {**os.environ, "HOME": "/data/claude", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "DISABLE_AUTOUPDATER": "1"}
    if ANTHROPIC_KEY:
        env["ANTHROPIC_API_KEY"] = ANTHROPIC_KEY
    if CLAUDE_TOKEN:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = CLAUDE_TOKEN
    return env


def job_public(j: dict) -> dict:
    end = j.get("finished") or time.time()
    return {"id": j["id"], "status": j["status"], "task": j["task"][:160], "allow_changes": j["allow_changes"],
            "seconds": int(end - j["started"]), "summary": (j.get("summary") or "")[:3000],
            "cost_usd": j.get("cost_usd"), "turns": j.get("turns")}


async def run_job(job: dict) -> None:
    Path("/data/claude").mkdir(parents=True, exist_ok=True)
    cfg = Path(f"/tmp/mcp-{job['id']}.json")
    cfg.write_text(json.dumps({"mcpServers": {"ha": {"type": "http", "url": MCP_URL}}}))
    sys_prompt = (
        "You are a Home Assistant operations agent working through the `ha` MCP tools only (no shell, "
        "no files). Be efficient: use ha_search / ha_get_overview to orient, make the smallest "
        "correct change, verify it by reading it back, and finish with a SHORT plain-spoken summary "
        "(2-4 sentences, no markdown) of what you found or changed. Never restart Home Assistant. "
        + ("You may modify configuration to complete the task." if job["allow_changes"]
           else "This job is READ-ONLY: do not change anything.")
    )
    cmd = ["claude", "-p", job["task"], "--output-format", "json", "--mcp-config", str(cfg),
           "--strict-mcp-config", "--allowedTools", "mcp__ha", "--max-turns", str(AGENT_TURNS),
           "--model", AGENT_MODEL, "--append-system-prompt", sys_prompt]
    deny = ["mcp__ha__ha_restart"]
    if not job["allow_changes"]:
        deny += [f"mcp__ha__{n}" for n in TOOL_NAMES if n.startswith(WRITE_PREFIXES)]
    cmd += ["--disallowedTools", *sorted(set(deny))]
    flog("agent_start", id=job["id"], task=job["task"], allow_changes=job["allow_changes"], model=AGENT_MODEL)
    log.info("agent %s start (changes=%s): %s", job["id"], job["allow_changes"], job["task"][:200])
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=agent_env(), cwd="/data/claude")
        try:
            out, err = await asyncio.wait_for(proc.communicate(), AGENT_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            job.update(status="timeout", summary=f"Agent timed out after {AGENT_TIMEOUT}s.")
            return
        text = out.decode("utf-8", "replace").strip()
        try:
            j = json.loads(text.splitlines()[-1] if text else "{}")
        except json.JSONDecodeError:
            j = {}
        if proc.returncode == 0 and j and not j.get("is_error"):
            job.update(status="done", summary=j.get("result") or "(no result)",
                       cost_usd=j.get("total_cost_usd"), turns=j.get("num_turns"))
        else:
            msg = j.get("result") or err.decode("utf-8", "replace")[-600:] or text[-600:] or f"exit {proc.returncode}"
            job.update(status="failed", summary=msg[:1500], turns=j.get("num_turns"))
    except FileNotFoundError:
        job.update(status="failed", summary="Claude Code CLI is not installed in this add-on image.")
    except Exception as exc:  # noqa: BLE001
        log.exception("agent job crashed")
        job.update(status="failed", summary=f"Agent crashed: {exc}")
    finally:
        job["finished"] = time.time()
        with contextlib.suppress(Exception):
            cfg.unlink()
        log.info("agent %s -> %s in %ds: %s", job["id"], job["status"],
                 job["finished"] - job["started"], (job.get("summary") or "")[:300])
        flog("agent_done", **job_public(job))


async def local_tool(name: str, args: dict) -> str:
    if name == "delegate_to_claude_agent":
        if not (ANTHROPIC_KEY or CLAUDE_TOKEN):
            return json.dumps({"error": "The Claude agent is not set up yet: add claude_oauth_token or "
                                        "anthropic_api_key in the add-on options."})
        task = str(args.get("task", "")).strip()
        if not task:
            return json.dumps({"error": "task is required"})
        running = [j for j in JOBS.values() if j["status"] == "running"]
        if len(running) >= 2:
            return json.dumps({"error": "Two agent jobs are already running; wait for one to finish."})
        job = {"id": uuid.uuid4().hex[:6], "task": task, "allow_changes": bool(args.get("allow_changes")),
               "status": "running", "started": time.time()}
        JOBS[job["id"]] = job
        asyncio.create_task(run_job(job))
        return json.dumps({"job_id": job["id"], "status": "running",
                           "note": "Started. Tell the user; the result will be announced when done."})
    if name == "check_claude_agent":
        jid = args.get("job_id")
        job = JOBS.get(jid) if jid else (max(JOBS.values(), key=lambda j: j["started"]) if JOBS else None)
        return json.dumps(job_public(job) if job else {"error": "no agent jobs yet"})
    return json.dumps({"error": "unknown local tool"})


# ---------------------------------------------------------------------- routes
@app.on_event("startup")
async def startup():
    flog("startup", model=MODEL, agent_model=AGENT_MODEL, agent_credentials=bool(ANTHROPIC_KEY or CLAUDE_TOKEN))
    try:
        p = await asyncio.create_subprocess_exec(
            "claude", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=agent_env())
        out, _ = await asyncio.wait_for(p.communicate(), 30)
        log.info("claude CLI: %s (credentials configured: %s)", out.decode().strip(),
                 bool(ANTHROPIC_KEY or CLAUDE_TOKEN))
    except Exception as exc:  # noqa: BLE001
        log.warning("claude CLI check failed: %r", exc)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/logs")
async def logs(n: int = 400):
    if not LOGFILE.exists():
        return PlainTextResponse("(no log yet)")
    lines = LOGFILE.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    return PlainTextResponse("\n".join(lines))


@app.get("/tools")
async def tools_view():
    if not TOOL_META and MCP_URL:
        with contextlib.suppress(Exception):
            await list_tools()
    out = []
    for m in TOOL_META:
        st = TOOL_STATS.get(m["name"], {})
        out.append({**m, "exposed": (not ALLOW) or m["name"] in ALLOW,
                    "writes": m["name"].startswith(WRITE_PREFIXES),
                    "confirm": is_destructive(m["name"], {}),
                    "calls": st.get("calls", 0), "errors": st.get("errors", 0),
                    "avg_ms": int(st["total_ms"] / st["calls"]) if st.get("calls") else None,
                    "last_ms": st.get("last_ms"), "last": st.get("last")})
    for lt in LOCAL_TOOLS:
        st = TOOL_STATS.get(lt["name"], {})
        props = lt["parameters"].get("properties", {})
        out.append({"name": lt["name"], "description": lt["description"], "local": True, "exposed": True,
                    "writes": lt["name"] == "delegate_to_claude_agent", "confirm": lt["name"] == "delegate_to_claude_agent" and CONFIRM_AGENT,
                    "params": [{"name": k, "type": v.get("type"), "required": k in lt["parameters"].get("required", []),
                                "description": v.get("description", "")} for k, v in props.items()],
                    "schema_bytes": len(json.dumps(lt["parameters"])), "calls": st.get("calls", 0), "errors": st.get("errors", 0),
                    "avg_ms": int(st["total_ms"] / st["calls"]) if st.get("calls") else None,
                    "last_ms": st.get("last_ms"), "last": st.get("last")})
    return {"tools": out, "total": len(TOOL_META), "exposed": sum(1 for t in out if t["exposed"]),
            "allowlist": sorted(ALLOW)}


@app.get("/logs.json")
async def logs_json(n: int = 600):
    rows = []
    if LOGFILE.exists():
        for line in LOGFILE.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]:
            with contextlib.suppress(Exception):
                rows.append(json.loads(line))
    return {"rows": rows}


@app.get("/agent/jobs")
async def agent_jobs():
    return {"jobs": [job_public(j) for j in sorted(JOBS.values(), key=lambda j: j["started"])]}


@app.get("/agent/selftest")
async def agent_selftest():
    p = await asyncio.create_subprocess_exec(
        "claude", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=agent_env())
    out, _ = await asyncio.wait_for(p.communicate(), 30)
    return {"claude_cli": out.decode().strip(), "credentials": bool(ANTHROPIC_KEY or CLAUDE_TOKEN),
            "mcp_url_set": bool(MCP_URL), "agent_model": AGENT_MODEL}


@app.post("/clientlog")
async def clientlog(request: Request):
    try:
        d = await request.json()
    except Exception:  # noqa: BLE001
        return {"ok": False}
    msg = str(d.get("msg", ""))[:800]
    log.info("browser: %s", msg)
    flog("browser", msg=msg)
    return {"ok": True}


@app.post("/session_end")
async def session_end(request: Request):
    try:
        d = await request.json()
    except Exception:  # noqa: BLE001
        return {"ok": False}
    rec = {k: d.get(k) for k in ("seconds", "cost_usd", "tokens", "model", "reason", "responses")}
    log.info("session end: %s", rec)
    flog("session_end", **rec)
    return {"ok": True}


@app.get("/sessions")
async def sessions(n: int = 15):
    out = []
    if LOGFILE.exists():
        for line in LOGFILE.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"kind": "session_end"' in line:
                with contextlib.suppress(Exception):
                    out.append(json.loads(line))
    return {"sessions": out[-n:][::-1]}


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
    tools = tools + LOCAL_TOOLS

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
    flog("session", model=MODEL, tools=len(tools), status=resp.status_code)
    if resp.status_code >= 400:
        log.error("client_secrets failed: %s %s", resp.status_code, resp.text[:300])
        return JSONResponse(
            {"error": f"OpenAI error {resp.status_code}: {resp.text[:300]}"}, 502
        )
    data = resp.json()
    return {"client_secret": data.get("value"), "tool_count": len(tools), "model": MODEL,
            "auto_stop": AUTO_STOP, "idle_timeout_s": IDLE_S, "prices": PRICES}


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
        if name == "delegate_to_claude_agent":
            summary = ("Start a Claude agent ("
                       + ("CAN CHANGE Home Assistant" if args.get("allow_changes") else "read-only")
                       + "):\n" + str(args.get("task", ""))[:500])
        else:
            summary = f"{name}({json.dumps(args)[:300]})"
        return {"needs_confirmation": True, "summary": summary}

    log.info("tool call: %s args=%s", name, json.dumps(args)[:300])
    st = TOOL_STATS.setdefault(name, {"calls": 0, "errors": 0, "total_ms": 0})
    flog("tool_call", name=name, args=json.dumps(args)[:600])
    tc0 = time.time()
    if name in ("delegate_to_claude_agent", "check_claude_agent"):
        text = await local_tool(name, args)
        ms0 = int((time.time() - tc0) * 1000)
        st.update(calls=st["calls"] + 1, total_ms=st["total_ms"] + ms0, last_ms=ms0, last=time.strftime("%H:%M:%S"))
        flog("tool_done", name=name, ms=int((time.time() - tc0) * 1000), out=text[:300])
        return {"output": text}
    try:
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
    except Exception as exc:  # noqa: BLE001
        log.exception("tool call failed")
        st.update(calls=st["calls"] + 1, errors=st["errors"] + 1, last=time.strftime("%H:%M:%S"))
        flog("tool_error", name=name, error=str(exc)[:300])
        return {"output": json.dumps({"error": str(exc)})}

    ms = int((time.time() - tc0) * 1000)
    st.update(calls=st["calls"] + 1, total_ms=st["total_ms"] + ms, last_ms=ms, last=time.strftime("%H:%M:%S"))
    if getattr(result, "is_error", getattr(result, "isError", False)):
        st["errors"] += 1
    log.info("tool done: %s in %d ms", name, ms)
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    flog("tool_done", name=name, ms=ms, out=text[:300])
    if len(text) > MAX_OUT:
        text = text[:MAX_OUT] + "\n...[truncated]"
    if getattr(result, "is_error", getattr(result, "isError", False)):
        text = json.dumps({"error": text})
    return {"output": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8099, log_level="info")
