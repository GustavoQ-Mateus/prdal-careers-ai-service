FROM python:3.12-slim
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
COPY apps/ai-service/requirements.txt ./
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
ARG EMBED_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"
COPY apps/ai-service/app ./app
COPY apps/ai-service/scripts ./scripts
RUN useradd --create-home --uid 10001 prdal && chown -R prdal:prdal /app
USER prdal
EXPOSE 8000
ENTRYPOINT ["python", "-m", "app.entrada"]
CMD ["app.lambda_handler.handler"]
