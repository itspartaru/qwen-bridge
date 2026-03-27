FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl bash && apt-get clean

# Устанавливаем Qwen CLI
RUN bash -c "$(curl -fsSL https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen.sh)" -s --source qwenchat || true

# Entrypoint загружает nvm (и node) перед запуском приложения
RUN printf '#!/bin/bash\nexport NVM_DIR="/root/.nvm"\n[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"\nexec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

RUN pip install fastapi uvicorn
COPY main.py .

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
