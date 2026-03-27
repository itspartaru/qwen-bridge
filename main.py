from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio, time, uuid, json

app = FastAPI()
POOL_SIZE = 3
workers: list["Worker"] = []


class Worker:
    def __init__(self):
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
        # Процесс ждёт ввода — system/init придёт с первым сообщением

    async def generate(self, text: str):
        """Генератор (delta, is_done, usage). Держит lock на время запроса."""
        msg = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]}
        }) + "\n"
        self.process.stdin.write(msg.encode())
        await self.process.stdin.drain()

        sent_len = 0
        while True:
            line = await self.process.stdout.readline()
            if not line:
                yield "", True, {}
                return
            raw = line.decode().strip()
            if not raw:
                continue
            data = json.loads(raw)
            t = data.get("type")

            if t == "system":
                continue  # повторный init между запросами — пропускаем

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

    async def is_alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def restart(self):
        try:
            self.process.kill()
        except Exception:
            pass
        await self.start()


async def get_worker() -> "Worker":
    """Ждёт свободного воркера, при необходимости перезапускает упавший."""
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
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


@app.on_event("startup")
async def startup():
    for _ in range(POOL_SIZE):
        w = Worker()
        await w.start()
        workers.append(w)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    do_stream = body.get("stream", False)
    prompt = format_prompt(messages)

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    worker = await get_worker()

    if do_stream:
        async def event_stream():
            async with worker.lock:
                async for delta, done, usage in worker.generate(prompt):
                    if done:
                        payload = {
                            "id": request_id, "object": "chat.completion.chunk",
                            "created": created, "model": "qwen-cli",
                            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        yield "data: [DONE]\n\n"
                    elif delta:
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
