ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache python3 py3-pip nodejs npm git ripgrep libgcc libstdc++ \
    && python3 -m venv /venv \
    && /venv/bin/pip install --no-cache-dir fastapi "uvicorn[standard]" httpx mcp \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

ENV USE_BUILTIN_RIPGREP=0
COPY app /app
WORKDIR /app

CMD ["/venv/bin/python", "/app/server.py"]
