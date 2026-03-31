from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio, time, uuid, json, logging, os, shlex

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen-bridge")

VERSION = "1.0.3"

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
TOOL_TO_EXEC: dict[str, "Callable[[dict], str]"] = {
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

# Active sessions keyed by tool_call_id for routing follow-up requests
active_sessions: dict[str, "Session"] = {}


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
        log.warning(f"[worker-{self.wid}] dead, restarting...")
        try:
            self.process.kill()
        except Exception:
            pass
        await self.start()


class Session:
    """
    One session = one qwen CLI conversation held for its full duration.
    Holds the worker lock until the conversation ends.

    out_q event types:
      {"type": "text",       "text": str}
      {"type": "tool_calls", "calls": [OpenAI tool_call, ...]}
      {"type": "done",       "usage": dict}
    """

    def __init__(self, session_id: str, worker: Worker):
        self.session_id = session_id
        self.worker = worker
        self.out_q: asyncio.Queue = asyncio.Queue()        # qwen CLI → HTTP handler
        self.tool_q: asyncio.Queue[list] = asyncio.Queue() # HTTP handler → qwen CLI
        self._active_ids: set[str] = set()
        self._pending_stubs: list[dict] = []  # stub results waiting to merge with pi-agent-core results
        self.task: asyncio.Task | None = None

    def _register(self, call_ids: list[str]):
        for cid in call_ids:
            active_sessions[cid] = self
        self._active_ids.update(call_ids)

    def _unregister(self):
        for cid in self._active_ids:
            active_sessions.pop(cid, None)
        self._active_ids.clear()

    async def run(self, prompt: str, token_timeout: float = 120.0):
        """Background task: drives the qwen CLI for the full conversation."""
        try:
            async with self.worker.lock:
                if not await self.worker.is_alive():
                    await self.worker.restart()

                await self._send({
                    "type": "user",
                    "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
                })

                while True:
                    event = await self._read_event(token_timeout)

                    if event["type"] == "done":
                        await self.out_q.put(event)
                        return

                    elif event["type"] == "tool_calls":
                        calls = event["calls"]
                        self._register([c["id"] for c in calls])
                        await self.out_q.put(event)

                        try:
                            tool_results = await asyncio.wait_for(
                                self.tool_q.get(), timeout=120.0
                            )
                        except asyncio.TimeoutError:
                            log.warning(f"[session-{self.session_id[:8]}] tool result timeout, killing worker")
                            try:
                                self.worker.process.kill()
                            except Exception:
                                pass
                            await self.out_q.put({"type": "done", "usage": {}})
                            return
                        finally:
                            self._unregister()

                        # Merge pi-agent-core results with any pre-computed stub results
                        all_results = self._pending_stubs + tool_results
                        self._pending_stubs = []

                        await self._send({
                            "type": "user",
                            "message": {"role": "user", "content": all_results},
                        })

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
                        # Real tools for pi-agent-core; stubs will be merged on return
                        self._pending_stubs = stub_results
                        names = [c["function"]["name"] for c in real_calls]
                        log.debug(f"[worker-{self.worker.wid}] tool_calls → pi-agent-core: {names}")
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
            if w.lock.locked():
                continue
            if not await w.is_alive():
                async with w.lock:
                    await w.restart()
            return w
        await asyncio.sleep(0.01)


# ─── Message formatting ───────────────────────────────────────────────────────

def format_prompt(messages: list, tools: list | None = None) -> str:
    parts = []

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

    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))

        if role == "tool":
            parts.append(f"tool_result [{m.get('tool_call_id', '')}]: {content}")
        elif role == "assistant" and m.get("tool_calls"):
            # Preserve tool call context so Qwen understands why tool_results follow
            call_strs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                call_strs.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
            suffix = f" [called: {'; '.join(call_strs)}]" if call_strs else ""
            parts.append(f"assistant:{suffix}")
        else:
            parts.append(f"{role}: {content}")

    return "\n".join(parts)


def find_session(messages: list) -> "Session | None":
    # Search from the end — most recent tool calls are at the bottom
    for m in reversed(messages):
        if m.get("role") == "tool":
            s = active_sessions.get(m.get("tool_call_id", ""))
            if s:
                return s
    return None


def extract_tool_results(messages: list) -> list:
    results = []
    for m in reversed(messages):
        if m.get("role") == "tool":
            results.insert(0, {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content", "") or "",
                "is_error": False,
            })
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            break
        else:
            break
    return results


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
    handed_off = False  # True when we paused for tool_calls — session stays live

    try:
        while True:
            try:
                item = await asyncio.wait_for(session.out_q.get(), timeout=180.0)
            except asyncio.TimeoutError:
                log.warning(f"[{request_id}] out_q timeout")
                break

            itype = item["type"]

            if itype == "text":
                text = item["text"]
                output += text
                if first_token is None:
                    first_token = time.monotonic() - t_start
                    log.info(f"[{request_id}] first token in {first_token:.2f}s")
                payload = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": created, "model": "qwen-cli",
                    "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}],
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
                handed_off = True
                yield "data: [DONE]\n\n"
                return  # pi-agent-core sends next request with tool results

            elif itype == "done":
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
        if not handed_off:
            if session.task and not session.task.done():
                session.task.cancel()
            session._unregister()


# ─── HTTP handler ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    log.warning(f"qwen-bridge v{VERSION} starting, pool={POOL_SIZE} workers")
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

    # ── Tool result continuation ──
    session = find_session(messages)
    if session:
        tool_results = extract_tool_results(messages)
        if tool_results:
            log.info(
                f"[{request_id}] → session-{session.session_id[:8]} "
                f"(tool results: {len(tool_results)})"
            )
            await session.tool_q.put(tool_results)

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
            return JSONResponse({
                "id": request_id, "object": "chat.completion",
                "created": created, "model": "qwen-cli",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0), "total_tokens": usage.get("total_tokens", 0)},
            })

    # ── New conversation ──
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
