FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    supervisor \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# ── CPU torch first — must be before requirements.txt ────────────────────
RUN pip install --no-cache-dir \
        "torch==2.2.2+cpu" \
        "torchvision==0.17.2+cpu" \
        --index-url https://download.pytorch.org/whl/cpu

# ── Everything else from pinned requirements ──────────────────────────────
COPY requirements_cpu.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements_cpu.txt

# ── Source code ───────────────────────────────────────────────────────────
COPY agents/ agents/
COPY api/ api/
COPY pipeline/ pipeline/
COPY ui/ ui/
COPY mlops/ mlops/
COPY mcp_server/ mcp_server/
COPY infra/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p data/processed data/xai data
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser data

ENV MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    XDG_CACHE_HOME=/tmp/cache

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
