# ai-service

Serviço Python de extração, geração, classificação e recuperação de contexto. Implementa a `spec-v1.11.0`.

## Instalação, testes e execução

Execute na raiz desta unidade. Não são necessários arquivos do monorepo. Requer Python 3.12.

```text
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt pytest
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m app.entrada
```

Em Linux use .venv/bin/python. Defina SERVICE_TOKEN com pelo menos 32 bytes aleatórios e PRDAL_AMBIENTE=desenvolvimento no uso local. ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL e AI_MODEL habilitam o provedor. Os testes usam clientes falsos; não exigem chave nem chamadas ao modelo. A imagem baixa o modelo EMBED_MODEL durante o build e usa cache próprio de embeddings. A porta HTTP é 8000; /health e /ready são as sondas.

## Imagem

```text
docker build -t prdal-ai-service .
```

O contexto é somente esta pasta. A imagem final executa sem root e não inclui dependências de desenvolvimento nem configurações de agentes. Injete as variáveis com --env-file em um arquivo local fora do controle de versão.

## Variáveis de ambiente

Use `.env.example` como referência, sem versionar segredos. As variáveis opcionais usam os padrões definidos no código; configure explicitamente os destinos de banco e serviços no seu ambiente.

`AI_CARACTERES_POR_TOKEN`, `AI_CONTAGEM_TOKENS`, `AI_JUIZ_RELACAO`, `AI_MAX_RETRIES`, `AI_MAX_TOKENS`, `AI_MODEL`, `AI_ORCAMENTO_ENTRADA_TOKENS`, `AI_TIMEOUT_PISO_S`, `AI_TIMEOUT_TETO_S`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `AWS_LAMBDA_RUNTIME_API`, `DOC_SERVICE_URL`, `EMBED_DIMENSAO`, `EMBED_MODEL`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `PRDAL_AMBIENTE`, `SERVICE_TOKEN`.
