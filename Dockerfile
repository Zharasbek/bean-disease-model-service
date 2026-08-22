FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/service/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /service

COPY requirements-api.txt /service/requirements-api.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /service/requirements-api.txt

# Verify that the important packages were installed.
RUN python -c "import torch, transformers, fastapi; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('fastapi:', fastapi.__version__)"

# Download the model into the Docker image.
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='Zharasbek/resnet18-bean-disease-classifier')"

COPY app /service/app

RUN useradd --create-home --uid 10001 apiuser \
    && chown -R apiuser:apiuser /service

USER apiuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
