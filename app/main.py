import os

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from .classify import classificar
from .copiloto import planejar_turno, redigir_formulario, redigir_mensagem
from .generate import generate_cv
from .keywords import extract_keywords
from .rag import consultar, indexar
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    GenerateCvRequest,
    GenerateCvResponse,
    IngestRequest,
    IngestResponse,
    KeywordsRequest,
    KeywordsResponse,
    QueryRequest,
    QueryResponse,
    RedigirFormularioRequest,
    RedigirFormularioResponse,
    RedigirMensagemRequest,
    RedigirMensagemResponse,
    ScoreRequest,
    ScoreResponse,
    TurnRequest,
    TurnResponse,
)
from .score import calcular_score

DOC_SERVICE_URL = os.getenv("DOC_SERVICE_URL", "http://localhost:8080")

app = FastAPI(title="ai-service")


class HealthResponse(BaseModel):
    service: str = "ai-service"
    status: str = "ok"


class HelloHop(BaseModel):
    service: str
    message: str


class HelloResponse(BaseModel):
    service: str
    message: str
    chain: list[HelloHop]


@app.on_event("startup")
def _warmup() -> None:
    try:
        from .rag import _model

        _model()
    except Exception:
        pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/hello", response_model=HelloResponse)
async def hello() -> HelloResponse:
    hop = HelloHop(service="ai-service", message="hello from ai-service")
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{DOC_SERVICE_URL}/hello")
        resp.raise_for_status()
        downstream = HelloResponse.model_validate(resp.json())
    return HelloResponse(
        service="ai-service",
        message=hop.message,
        chain=[hop, *downstream.chain],
    )


@app.post("/keywords", response_model=KeywordsResponse)
def keywords(req: KeywordsRequest) -> KeywordsResponse:
    return KeywordsResponse(keywords=extract_keywords(req.descricao))


@app.post("/generate-cv", response_model=GenerateCvResponse)
def generate(req: GenerateCvRequest) -> GenerateCvResponse:
    return GenerateCvResponse(markdown=generate_cv(req))


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    return calcular_score(req.markdown, req.vaga.keywords)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    return classificar(req.titulo, req.descricao)


@app.post("/context/ingest", response_model=IngestResponse)
def context_ingest(req: IngestRequest) -> IngestResponse:
    return indexar(req.documentos)


@app.post("/context/query", response_model=QueryResponse)
def context_query(req: QueryRequest) -> QueryResponse:
    return consultar(req.usuario_id, req.query, req.k)


@app.post("/copiloto/turn", response_model=TurnResponse)
def copiloto_turn(req: TurnRequest) -> TurnResponse:
    return planejar_turno(req)


@app.post("/copiloto/redigir-mensagem", response_model=RedigirMensagemResponse)
def copiloto_redigir_mensagem(
    req: RedigirMensagemRequest,
) -> RedigirMensagemResponse:
    return redigir_mensagem(req)


@app.post("/copiloto/redigir-formulario", response_model=RedigirFormularioResponse)
def copiloto_redigir_formulario(
    req: RedigirFormularioRequest,
) -> RedigirFormularioResponse:
    return redigir_formulario(req)
