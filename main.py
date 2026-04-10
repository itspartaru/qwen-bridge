from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio, time, uuid, json, logging, os, shlex, re
from typing import Callable

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen-bridge")

VERSION = "1.0.9"

# ─── Artifact filter ──────────────────────────────────────────────────────────
# Qwen Code echoes the format_prompt history format back into its own text
# output (e.g. "[called: exec({...})]", "tool_result [callXXX]: ...").
# This filter strips those lines so they don't reach OpenClaw / Telegram.

_ARTIFACT_RE = re.compile(
    r'^\[called:|^tool_result \[|^assistant:\s+\[called:|^<tool_call:|^<tool_result>'
)


class _ArtifactFilter:
    """Buffer streaming text tokens and suppress Qwen artifact lines."""

    def __init__(self):
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        lines = self._buf.split("\n")
        self._buf = lines[-1]          # keep the incomplete trailing fragment
        out = []
        for line in lines[:-1]:
            if not _ARTIFACT_RE.match(line):
                out.append(line)
        return "\n".join(out) + ("\n" if len(lines) > 1 else "")

    def flush(self) -> str:
        text = self._buf
        self._buf = ""
        return "" if _ARTIFACT_RE.match(text) else text


def _strip_artifacts(text: str) -> str:
    """Remove Qwen artifact lines from a completed text block."""
    lines = text.splitlines(keepends=True)
    return "".join(l for l in lines if not _ARTIFACT_RE.match(l))
CONTEXT_MESSAGES_LIMIT = int(os.getenv("CONTEXT_MESSAGES_LIMIT", "20"))

# ─── Tool mapping: Qwen Code ↔ pi-agent-core ─────────────────────────────────
#
# pi-agent-core executes tools in the OpenClaw container.
# Qwen Code uses its own internal tool names — we translate them here.

# Qwen Code tool name → pi-agent-core tool name
TOOL_NAME_MAP: dict[str, str] = {
    "run_shell_command": "exec",
    "read_file":         "read",
    "write_file":        "write",
    "edit":              "edit",        # same name
    "web_search":        "web_search",  # same name
    "web_fetch":         "web_fetch",   # same name
}

# Argument key remapping per pi-agent-core tool name: {qwen_arg: pi_arg}
TOOL_ARG_MAP: dict[str, dict[str, str]] = {
    "read":  {"file_path": "path"},
    "write": {"file_path": "path"},
    "edit":  {"file_path": "path"},
}

# Qwen Code tools that have no direct equivalent — map to exec with generated command
TOOL_TO_EXEC: dict[str, Callable[[dict], str]] = {
    "list_directory": lambda i: f"ls -la {shlex.quote(i.get('path', i.get('directory', '.')))}",
    "glob":           lambda i: f"find . -name {shlex.quote(i.get('pattern', '*'))}",
    "grep_search":    lambda i: (
        f"grep -rn {shlex.quote(i.get('query', i.get('pattern', '')))} "
        f"{shlex.quote(i.get('path', i.get('directory', '.')))}"
    ),
}

# Qwen Code tools with no pi-agent-core equivalent — stub result returned inline
# (Qwen Code will see these as successful and continue without involving pi-agent-core)
TOOL_STUBS: dict[str, str] = {
    "save_memory":       "Memory saved.",
    "todo_write":        "Todo updated.",
    "skill":             "Skill invoked.",
    "agent":             "Agent task delegated.",
    "ask_user_question": "ok",
}

# ─────────────────────────────────────────────────────────────────────────────


def remap_tool(name: str, input_args: dict) -> tuple[str, dict]:
    """Translate a Qwen Code tool call to pi-agent-core name + args."""
    if name in TOOL_TO_EXEC:
        return "exec", {"command": TOOL_TO_EXEC[name](input_args)}

    mapped_name = TOOL_NAME_MAP.get(name, name)
    arg_remap = TOOL_ARG_MAP.get(mapped_name, {})
    mapped_args = {arg_remap.get(k, k): v for k, v in input_args.items()}
    return mapped_name, mapped_args


app = FastAPI()
POOL_SIZE = 3
workers: list["Worker"] = []


class Worker:
    def __init__(self, wid: int):
        self.wid = wid
        self.process = None
        self.lock = asyncio.Lock()

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            "qwen",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--exclude-tools", "skill",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info(f"[worker-{self.wid}] started (pid={self.process.pid})")

    async def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def restart(self):
        log.warning(f"[worker-{self.wid}] restarting...")
        try:
            self.process.kill()
        except Exception:
            pass
        await self.start()


class Session:
    """
    Stateless session: one qwen CLI interaction per HTTP request.
    Worker is always restarted at the start to ensure clean context.
    Tool results from openclaw are NOT waited for — when qwen issues real
    tool calls they are returned immediately and the session ends.
    The next request will carry the full message history (including tool
    results) which format_prompt renders into a single prompt for qwen.

    out_q event types:
      {"type": "text",       "text": str}
      {"type": "tool_calls", "calls": [OpenAI tool_call, ...]}
      {"type": "done",       "usage": dict}
    """

    def __init__(self, session_id: str, worker: Worker):
        self.session_id = session_id
        self.worker = worker
        self.out_q: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    async def run(self, prompt: str, token_timeout: float = 120.0):
        """Background task: drives the qwen CLI for this request."""
        try:
            async with self.worker.lock:
                # Always restart — stateless mode requires a fresh qwen context
                await self.worker.restart()

                await self._send({
                    "type": "user",
                    "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
                })

                while True:
                    event = await self._read_event(token_timeout)
                    await self.out_q.put(event)
                    if event["type"] in ("done", "tool_calls"):
                        return

        except asyncio.CancelledError:
            log.warning(f"[session-{self.session_id[:8]}] cancelled")
            try:
                self.worker.process.kill()
            except Exception:
                pass
            raise

    async def _send(self, obj: dict):
        line = json.dumps(obj) + "\n"
        self.worker.process.stdin.write(line.encode())
        await self.worker.process.stdin.drain()

    async def _read_event(self, token_timeout: float) -> dict:
        """Read qwen CLI stdout until a complete logical event is ready."""
        pending_tools: list[dict] = []
        usage_buf: dict = {}

        while True:
            try:
                line = await asyncio.wait_for(
                    self.worker.process.stdout.readline(),
                    timeout=token_timeout,
                )
            except asyncio.TimeoutError:
                stderr_snippet = ""
                try:
                    raw_err = await asyncio.wait_for(
                        self.worker.process.stderr.read(4096), timeout=1.0
                    )
                    stderr_snippet = raw_err.decode(errors="replace").strip()
                except Exception:
                    pass
                log.warning(
                    f"[worker-{self.worker.wid}] readline timeout after {token_timeout}s, killing"
                    + (f" | stderr: {stderr_snippet[:300]}" if stderr_snippet else "")
                )
                try:
                    self.worker.process.kill()
                except Exception:
                    pass
                return {"type": "done", "usage": {}}

            if not line:
                return {"type": "done", "usage": {}}

            raw = line.decode().strip()
            if not raw:
                continue

            log.debug(f"[worker-{self.worker.wid}] ← {raw}")
            data = json.loads(raw)
            t = data.get("type")

            if t == "system":
                continue

            elif t == "stream_event":
                event = data.get("event", {})
                et = event.get("type")

                if et == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            await self.out_q.put({"type": "text", "text": text})

                elif et == "message_stop" and pending_tools:
                    stub_results: list[dict] = []
                    real_calls: list[dict] = []

                    for tu in pending_tools:
                        name = tu["name"]
                        if name in TOOL_STUBS:
                            stub_results.append({
                                "type": "tool_result",
                                "tool_use_id": tu["id"],
                                "content": TOOL_STUBS[name],
                                "is_error": False,
                            })
                        else:
                            mapped_name, mapped_args = remap_tool(name, tu.get("input", {}))
                            real_calls.append({
                                "id": tu["id"],
                                "type": "function",
                                "function": {
                                    "name": mapped_name,
                                    "arguments": json.dumps(mapped_args),
                                },
                            })
                    pending_tools.clear()

                    if not real_calls:
                        # All stubs — inject directly and continue reading
                        log.debug(
                            f"[worker-{self.worker.wid}] stub-only tool_calls, injecting inline"
                        )
                        await self._send({
                            "type": "user",
                            "message": {"role": "user", "content": stub_results},
                        })
                        # Continue the while loop — don't return a tool_calls event
                    else:
                        # Real tools for pi-agent-core; session ends here.
                        # Any stub results from this batch will be synthesized by
                        # format_prompt when openclaw sends the next request with history.
                        names = [c["function"]["name"] for c in real_calls]
                        log.debug(f"[worker-{self.worker.wid}] tool_calls → openclaw: {names}")
                        return {"type": "tool_calls", "calls": real_calls}

            elif t == "assistant":
                msg = data.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") == "tool_use":
                        pending_tools.append(block)
                        log.debug(
                            f"[worker-{self.worker.wid}] tool_use: {block['name']}({block['id']})"
                        )
                last_usage = msg.get("usage", {})
                if last_usage:
                    usage_buf = last_usage

            elif t == "result":
                return {"type": "done", "usage": data.get("usage", usage_buf)}


# ─── Worker pool ──────────────────────────────────────────────────────────────

async def get_worker() -> Worker:
    while True:
        for w in workers:
            if not w.lock.locked():
                return w
        await asyncio.sleep(0.01)


# ─── Message formatting ───────────────────────────────────────────────────────

def format_prompt(messages: list, tools: list | None = None) -> str:
    parts = [
        "[HISTORY — do not reproduce <tool_call:>, <tool_result>, or [called:] "
        "tags in your response. Reply naturally as the assistant.]"
    ]

    # Inject pi-agent-core tool definitions so Qwen Code knows about them
    if tools:
        lines = ["[Additional tools available — call these by name when needed]"]
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            props = fn.get("parameters", {}).get("properties", {})
            req = fn.get("parameters", {}).get("required", [])
            params = []
            for k, v in props.items():
                flag = "" if k in req else "?"
                params.append(f"{k}{flag}: {v.get('type', 'any')}")
            params_str = f"({', '.join(params)})" if params else "()"
            lines.append(f"  {name}{params_str} — {desc}")
        parts.append("\n".join(lines))

    # Deduplicate system messages by content (openclaw resends the full history
    # on every tool-result round-trip, so the same system prompt appears repeatedly)
    seen_system: set[str] = set()
    system_msgs = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "") or ""
            if content not in seen_system:
                seen_system.add(content)
                system_msgs.append(m)

    # Sliding window: always keep (deduplicated) system messages; limit non-system to last N
    other_msgs = [m for m in messages if m.get("role") != "system"]
    if CONTEXT_MESSAGES_LIMIT > 0:
        other_msgs = other_msgs[-CONTEXT_MESSAGES_LIMIT:]
    windowed = system_msgs + other_msgs

    # Pre-collect tool_call IDs that already have results in the history.
    # Stub tool calls from a previous turn may be missing results because the
    # session ended before they were merged — we synthesize them below.
    covered_ids: set[str] = {
        m.get("tool_call_id", "") for m in windowed if m.get("role") == "tool"
    }

    for m in windowed:
        role = m.get("role", "")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))

        if role == "tool":
            # Neutral XML wrapper — avoids teaching the model to output "tool_result [id]:" text
            parts.append(f"<tool_result>{content}</tool_result>")

        elif role == "assistant" and m.get("tool_calls"):
            # Show text content separately; tool calls as XML so the model
            # doesn't learn to reproduce "[called: name({...})]" in its own output
            if content:
                parts.append(f"assistant: {content}")
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                parts.append(f"<tool_call:{fn.get('name', '')}>{fn.get('arguments', '')}</tool_call:{fn.get('name', '')}>")

            # Synthesize missing stub results so qwen sees a complete tool-result batch
            for tc in m["tool_calls"]:
                tc_id = tc.get("id", "")
                fn_name = tc.get("function", {}).get("name", "")
                if tc_id and tc_id not in covered_ids and fn_name in TOOL_STUBS:
                    parts.append(f"<tool_result>{TOOL_STUBS[fn_name]}</tool_result>")

        else:
            parts.append(f"{role}: {content}")

    return "\n".join(parts)


# ─── Logging helpers ──────────────────────────────────────────────────────────

def _fmt(text: str, limit: int) -> str:
    text = text.replace("\n", " ")
    if log.isEnabledFor(logging.DEBUG):
        return text
    return text if len(text) <= limit else text[:limit] + "…"


def log_request(rid: str, messages: list, do_stream: bool):
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    text = ""
    if last:
        c = last.get("content") or ""
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        text = c
    log.info(f"[{rid}] ▶ stream={do_stream} msgs={len(messages)} | user: {_fmt(text, 120)!r}")


# ─── Streaming ────────────────────────────────────────────────────────────────

async def _iter_session(session: Session, request_id: str, created: int):
    t_start = time.monotonic()
    first_token: float | None = None
    output = ""
    af = _ArtifactFilter()

    try:
        while True:
            try:
                item = await asyncio.wait_for(session.out_q.get(), timeout=180.0)
            except asyncio.TimeoutError:
                log.warning(f"[{request_id}] out_q timeout")
                break

            itype = item["type"]

            if itype == "text":
                clean = af.feed(item["text"])
                output += clean
                if clean:
                    if first_token is None:
                        first_token = time.monotonic() - t_start
                        log.info(f"[{request_id}] first token in {first_token:.2f}s")
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk",
                        "created": created, "model": "qwen-cli",
                        "choices": [{"delta": {"content": clean}, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            elif itype == "tool_calls":
                calls = item["calls"]
                log.info(f"[{request_id}] → tool_calls: {[c['function']['name'] for c in calls]}")
                for i, call in enumerate(calls):
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk",
                        "created": created, "model": "qwen-cli",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "index": i,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": call["function"]["arguments"],
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                payload = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": created, "model": "qwen-cli",
                    "choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                return

            elif itype == "done":
                # Flush any buffered fragment from the artifact filter
                tail = af.flush()
                if tail:
                    output += tail
                    payload = {
                        "id": request_id, "object": "chat.completion.chunk",
                        "created": created, "model": "qwen-cli",
                        "choices": [{"delta": {"content": tail}, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                usage = item.get("usage", {})
                elapsed = time.monotonic() - t_start
                tokens = f"in={usage.get('input_tokens','?')} out={usage.get('output_tokens','?')}"
                log.info(f"[{request_id}] ✓ {elapsed:.2f}s {tokens} | {_fmt(output, 200)!r}")
                payload = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": created, "model": "qwen-cli",
                    "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                return

    finally:
        if session.task and not session.task.done():
            session.task.cancel()


# ─── HTTP handler ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    log.warning(f"qwen-bridge v{VERSION} starting, pool={POOL_SIZE} workers, CONTEXT_MESSAGES_LIMIT={CONTEXT_MESSAGES_LIMIT}")
    for i in range(POOL_SIZE):
        w = Worker(wid=i)
        await w.start()
        workers.append(w)
    log.info("All workers ready")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools") or None
    do_stream = body.get("stream", False)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    log_request(request_id, messages, do_stream)

    # Stateless: every request is a fresh conversation.
    # format_prompt renders the full message history (including past tool calls
    # and results from openclaw) into a single prompt for qwen.
    prompt = format_prompt(messages, tools=tools)
    worker = await get_worker()
    session = Session(session_id=uuid.uuid4().hex, worker=worker)
    log.info(f"[{request_id}] → worker-{worker.wid} session-{session.session_id[:8]}")
    session.task = asyncio.create_task(session.run(prompt))

    if do_stream:
        return StreamingResponse(
            _iter_session(session, request_id, created),
            media_type="text/event-stream",
        )

    output, usage = "", {}
    while True:
        item = await asyncio.wait_for(session.out_q.get(), timeout=180.0)
        if item["type"] == "text":
            output += item["text"]
        elif item["type"] == "tool_calls":
            return JSONResponse({
                "id": request_id, "object": "chat.completion",
                "created": created, "model": "qwen-cli",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": item["calls"]},
                    "finish_reason": "tool_calls",
                }],
            })
        elif item["type"] == "done":
            usage = item.get("usage", {})
            break

    output = _strip_artifacts(output)
    return JSONResponse({
        "id": request_id, "object": "chat.completion",
        "created": created, "model": "qwen-cli",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", len(prompt) // 4),
            "completion_tokens": usage.get("output_tokens", len(output) // 4),
            "total_tokens": usage.get("total_tokens", (len(prompt) + len(output)) // 4),
        },
    })
