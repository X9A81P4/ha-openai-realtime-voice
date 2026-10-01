ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache python3 py3-pip \
    && python3 -m venv /venv \
    && /venv/bin/pip install --no-cache-dir fastapi "uvicorn[standard]" httpx mcp

COPY app /app
WORKDIR /app

CMD ["/venv/bin/python", "/app/server.py"]
