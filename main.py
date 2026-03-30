from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio, time, uuid, json, logging

import os
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen-bridge")

VERSION = "0.95.3"

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
            "--exclude-tools", "agent", "skill", "list_directory", "read_file",
            "grep_search", "glob", "edit", "write_file", "run_shell_command",
            "save_memory", "todo_write", "ask_user_question", "exit_plan_mode",
            "web_fetch", "web_search",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.info(f"[worker-{self.wid}] started (pid={self.process.pid})")

    async def generate(self, text: str, token_timeout: float = 120.0):
        """Генератор (delta, is_done, usage). token_timeout — таймаут между токенами."""
        msg = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]}
        }) + "\n"
        self.process.stdin.write(msg.encode())
        await self.process.stdin.drain()

        sent_len = 0
        usage_buf = {}
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        timeout=token_timeout,
                    )
                except asyncio.TimeoutError:
                    # Читаем stderr чтобы понять что происходило
                    stderr_out = ""
                    try:
                        stderr_out = await asyncio.wait_for(
                            self.process.stderr.read(4096), timeout=1.0
                        )
                        stderr_out = stderr_out.decode(errors="replace").strip()
                    except Exception:
                        pass
                    log.warning(
                        f"[worker-{self.wid}] readline timeout after {token_timeout}s, killing process"
                        + (f" | stderr: {stderr_out[:500]}" if stderr_out else "")
                    )
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                    yield "", True, {}
                    return
                if not line:
                    yield "", True, {}
                    return
                raw = line.decode().strip()
                if not raw:
                    continue
                log.debug(f"[worker-{self.wid}] ← {raw[:300]}")
                data = json.loads(raw)
                t = data.get("type")

                if t == "system":
                    continue
                elif t == "stream_event":
                    event = data.get("event", {})
                    et = event.get("type")
                    if et == "content_block_delta":
                        delta_obj = event.get("delta", {})
                        if delta_obj.get("type") == "text_delta":
                            text = delta_obj.get("text", "")
                            if text:
                                yield text, False, {}
                elif t == "assistant":
                    # последнее assistant-сообщение содержит финальный usage
                    last_usage = data.get("message", {}).get("usage", {})
                    if last_usage:
                        usage_buf = last_usage
                elif t == "result":
                    yield "", True, data.get("usage", usage_buf)
                    return
        except asyncio.CancelledError:
            # Клиент отключился — убиваем процесс чтобы не оставлять мусор в stdout
            log.warning(f"[worker-{self.wid}] client disconnected, killing process")
            try:
                self.process.kill()
            except Exception:
                pass
            raise

    async def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def restart(self):
        log.warning(f"[worker-{self.wid}] dead, restarting...")
        try:
            self.process.kill()
        except Exception:
            pass
        await self.start()


async def get_worker() -> "Worker":
    while True:
        for w in workers:
            if w.lock.locked():
                continue
            if not await w.is_alive():
                async with w.lock:
                    await w.restart()
            return w
        await asyncio.sleep(0.01)


def format_prompt(messages):
    parts = []
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        parts.append(f"{m['role']}: {content}")
    return "\n".join(parts)


def _fmt(text: str, limit: int) -> str:
    text = text.replace("\n", " ")
    if log.isEnabledFor(logging.INFO):
        return text
    return text[:limit] + "…" if len(text) > limit else text


def log_request(rid: str, messages: list, do_stream: bool):
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    text = ""
    if last:
        c = last.get("content") or ""
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        text = c
    msg = f"[{rid}] ▶ stream={do_stream} msgs={len(messages)} | user: {_fmt(text, 120)!r}"
    if log.isEnabledFor(logging.INFO):
        log.info(msg)
    else:
        log.warning(msg)


def log_response(rid: str, output: str, usage: dict, elapsed: float):
    tokens = f"in={usage.get('input_tokens','?')} out={usage.get('output_tokens','?')} total={usage.get('total_tokens','?')}"
    msg = f"[{rid}] ✓ {elapsed:.2f}s {tokens} | reply: {_fmt(output, 200)!r}"
    if log.isEnabledFor(logging.INFO):
        log.info(msg)
    else:
        log.warning(msg)


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
    do_stream = body.get("stream", False)
    prompt = format_prompt(messages)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    log_request(request_id, messages, do_stream)
    t_start = time.monotonic()

    worker = await get_worker()
    log.info(f"[{request_id}] → worker-{worker.wid}")

    if do_stream:
        async def event_stream():
            output = ""
            usage = {}
            first_token = None
            async with worker.lock:
                async for delta, done, u in worker.generate(prompt):
                    if done:
                        usage = u
                        elapsed = time.monotonic() - t_start
                        log_response(request_id, output, usage, elapsed)
                        payload = {
                            "id": request_id, "object": "chat.completion.chunk",
                            "created": created, "model": "qwen-cli",
                            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        yield "data: [DONE]\n\n"
                    elif delta:
                        if first_token is None:
                            first_token = time.monotonic() - t_start
                            log.info(f"[{request_id}] first token in {first_token:.2f}s")
                        output += delta
                        log.debug(f"[{request_id}] ▸ {delta!r}")
                        payload = {
                            "id": request_id, "object": "chat.completion.chunk",
                            "created": created, "model": "qwen-cli",
                            "choices": [{"delta": {"content": delta}, "index": 0, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        output = ""
        usage = {}
        async with worker.lock:
            async for delta, done, u in worker.generate(prompt):
                if done:
                    usage = u
                else:
                    output += delta

        elapsed = time.monotonic() - t_start
        log_response(request_id, output, usage, elapsed)

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": "qwen-cli",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", len(prompt) // 4),
                "completion_tokens": usage.get("output_tokens", len(output) // 4),
                "total_tokens": usage.get("total_tokens", (len(prompt) + len(output)) // 4)
            }
        })
