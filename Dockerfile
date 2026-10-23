FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY tests ./tests

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

# Self-sufficient default: seed the catalog, then serve. Cloudflare Containers
# run this CMD directly, with no compose file to layer a command on top.
#
# --proxy-headers makes uvicorn honour X-Forwarded-Proto. Without it Starlette's
# url_for builds absolute http:// URLs for CSS and JS from the request scheme,
# and a browser on the https site blocks them as mixed content. Only Cloudflare
# can reach this port, so the forwarded header is trusted.
CMD ["sh", "-c", "python -m vramsherpa.seed && exec uvicorn vramsherpa.main:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips=*"]
