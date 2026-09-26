# Transfer Edge — Hugging Face Space image.
# The app only, with the data and models it reads baked in (a Space has no
# mounted data/ folder). Assembled into build/hf_space/ by deploy/build_space.py.
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
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

# Spaces run the container as UID 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
USER user

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
