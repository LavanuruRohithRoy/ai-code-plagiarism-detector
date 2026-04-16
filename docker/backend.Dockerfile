FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p /app/db /app/data/raw /app/data/embeddings /app/hf_cache

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
	python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
	grep -iv '^torch$' requirements.txt > requirements.docker.txt && \
	python -m pip install --no-cache-dir -r requirements.docker.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Pre-cache CodeBERT model during build to avoid download delays on startup
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
	AutoTokenizer.from_pretrained('microsoft/codebert-base'); \
	AutoModel.from_pretrained('microsoft/codebert-base')"

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
