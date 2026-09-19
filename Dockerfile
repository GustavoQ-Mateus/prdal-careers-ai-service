FROM python:3.12-slim
WORKDIR /app
COPY apps/ai-service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ARG EMBED_MODEL=paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"
COPY apps/ai-service/app ./app
COPY apps/ai-service/scripts ./scripts
COPY .claude/agents ./.claude/agents
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
