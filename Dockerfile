FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAGKIT_ENV=prod RAGKIT_HOME=/app
WORKDIR /app

# EXTRAS="" for the API image, EXTRAS=ui for the Streamlit image (docker-compose sets it)
ARG EXTRAS=""
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"

COPY configs ./configs
COPY data ./data
COPY eval ./eval
COPY .streamlit ./.streamlit

# run as non-root
RUN useradd --system --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/readyz')"
# PSEUDO_KEY (required in prod), RAGKIT_ADMIN_TOKEN and ANTHROPIC_API_KEY are injected at runtime, never baked in.
CMD ["uvicorn", "--factory", "ragkit.api.app:get_app", "--host", "0.0.0.0", "--port", "8000"]
