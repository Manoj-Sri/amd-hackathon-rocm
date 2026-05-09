# Hugging Face Space Dockerfile for GPU Goblin (Streamlit UI).
#
# Why Docker (vs. HF's `sdk: streamlit`): the HF web UI's new-space picker
# doesn't expose Streamlit as a first-class SDK choice anymore (only
# Gradio / Docker / Static are listed). Docker SDK is the cleanest way to
# keep our Streamlit UI exactly as-is without migrating to Gradio.
#
# Build behavior matches the offline-replay-only Space:
#   * No torch / transformers / sentence-transformers — we ship the KB
#     embeddings cache and `ui/app.py` only imports light deps.
#   * Cold-start ~30-60s on HF's CPU-basic tier.
#   * Default Streamlit port is 7860 on HF (NOT 8501 — HF expects 7860).

FROM python:3.11-slim

# HF Spaces require the app to bind to 0.0.0.0:7860.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    # HF Spaces serves Streamlit inside an iframe at *.hf.space while the
    # parent page lives at huggingface.co/spaces. The browser drops Streamlit's
    # XSRF cookie on cross-origin XHRs, so st.file_uploader gets a 403 from
    # /_stcore/upload_file (surfaces in the UI as `AxiosError 403`). CORS
    # check has the same root cause. Disable both — there's no untrusted
    # origin in this deployment shape anyway.
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# HF Spaces convention: code lives at /app, owned by a non-root user.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# Copy requirements first to maximize Docker layer caching.
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app.
COPY --chown=user:user . /app

USER user

# HF expects the container to listen on 7860.
EXPOSE 7860

# Streamlit entry point. ui/app.py self-bootstraps the repo root onto
# sys.path so `from agent.schemas import ...` resolves.
CMD ["streamlit", "run", "ui/app.py"]
