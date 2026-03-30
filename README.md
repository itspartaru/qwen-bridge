# qwen-bridge

OpenAI-compatible API bridge for [Qwen Code](https://qwen.ai) CLI. Wraps the Qwen CLI in a FastAPI server so any OpenAI-compatible client can talk to Qwen.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Real streaming via Server-Sent Events
- Persistent worker pool (3 Qwen processes) — no Node.js startup overhead per request
- Docker image with Qwen CLI pre-installed

## Quick start

### 1. Pull and run

```bash
docker run -d \
  --name qwen-bridge \
  -p 5000:5000 \
  -v /your/path/docker-config:/root/.config \
  -v /your/path/docker-qwen:/root/.qwen \
  --restart unless-stopped \
  itsparta/qwen-bridge:latest
```

Or with Docker Compose — see [docker-compose.yml](docker-compose.yml).

### 2. Authorize Qwen

```bash
docker exec -it qwen-bridge bash
source /root/.nvm/nvm.sh
qwen auth login
```

Auth is persisted via the mounted `/root/.qwen` volume — survives container restarts.

### 3. Test

```bash
# Non-streaming
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

# Streaming
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Portainer stack

```yaml
services:
  qwen-bridge:
    image: itsparta/qwen-bridge:latest
    container_name: qwen-bridge
    ports:
      - "5000:5000"
    volumes:
      - /your/path/docker-config:/root/.config
      - /your/path/docker-qwen:/root/.qwen
    restart: unless-stopped
```

## Logging

Controlled by the `LOG_LEVEL` environment variable:

| `LOG_LEVEL`         | Behavior                                                     |
|---------------------|--------------------------------------------------------------|
| `WARNING` (default) | request preview (120 chars) + response preview (200 chars)   |
| `INFO`              | full request and response text                               |
| `DEBUG`             | full text + every streaming chunk                            |

Set in `docker-compose.yml` or via `docker run -e LOG_LEVEL=DEBUG ...`.

## How it works

On startup, the bridge pre-launches 3 persistent Qwen CLI processes using `--input-format stream-json`. Each request acquires a free worker via `asyncio.Lock`, sends the message over stdin, and streams the response back. Workers are automatically restarted on failure.

## Tech stack

- Python 3.11, FastAPI, Uvicorn
- Qwen Code CLI 0.13.1
- Docker
