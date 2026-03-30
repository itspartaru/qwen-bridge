from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio, time, uuid, json, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qwen-bridge")

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
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        timeout=token_timeout,
                    )
                except asyncio.TimeoutError:
                    log.warning(f"[worker-{self.wid}] readline timeout after {token_timeout}s, killing process")
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
                data = json.loads(raw)
                t = data.get("type")

                if t == "system":
                    continue
                elif t == "assistant":
                    for block in data.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            full = block["text"]
                            delta = full[sent_len:]
                            if delta:
                                sent_len = len(full)
                                yield delta, False, {}
                elif t == "result":
                    yield "", True, data.get("usage", {})
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


def log_request(rid: str, messages: list, do_stream: bool):
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    preview = ""
    if last:
        c = last.get("content") or ""
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        preview = c.replace("\n", " ")
    log.info(f"[{rid}] ▶ stream={do_stream} msgs={len(messages)} | user: {preview!r}")


def log_response(rid: str, output: str, usage: dict, elapsed: float):
    preview = output.replace("\n", " ")
    tokens = f"in={usage.get('input_tokens','?')} out={usage.get('output_tokens','?')} total={usage.get('total_tokens','?')}"
    log.info(f"[{rid}] ✓ {elapsed:.2f}s {tokens} | reply: {preview!r}")


@app.on_event("startup")
async def startup():
    log.info(f"Starting {POOL_SIZE} workers...")
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
