# Transfer Edge — one image for the app, training jobs and the notebooks.
# data/ and models/ are NOT baked in: docker-compose.yml mounts them, so the
# container and the local environment share the same files.
FROM python:3.13-slim

# libgomp1: the OpenMP runtime xgboost links against; slim images don't ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    # notebooks locate the project from this instead of walking up to CLAUDE.md
    TRANSFER_EDGE_ROOT=/app \
    # shap imports numba, and matplotlib writes a font cache: both need a writable dir
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501 8888
CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
