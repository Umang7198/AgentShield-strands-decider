# ShopHelp in one image: FastAPI backend, desk UI, SQLite.
# The 2B Decider is Apple MLX and stays on the host; this image uses
# FAKE_DECIDER=1 by default (policy still gates). Point DECIDER_URL at
# host.docker.internal:8099 and set FAKE_DECIDER=0 to use a host Decider.
#
#   docker compose up --build
#   open http://127.0.0.1:8080/
#
# Pip runs once, keyed only on pyproject.toml. App files are copied afterward
# with --chown so a src/frontend change does not reinstall wheels.

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8080 \
    FAKE_DECIDER=1 \
    DATABASE_URL=sqlite:////data/agentshield.db

COPY pyproject.toml ./
RUN mkdir -p src/agentshield \
    && printf '"""stub for docker dep cache"""\n' > src/agentshield/__init__.py \
    && pip install --no-cache-dir --retries 5 -e ".[strands]" \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /data

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser policies ./policies
COPY --chown=appuser:appuser app.py ./

USER appuser
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=25s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"

CMD ["python", "app.py", "--no-browser"]
