import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .carregador_prompts import carregar_prompts
from .classify import classificar, taxonomia
from .copiloto import planejar_turno, redigir_formulario, redigir_mensagem
from .degradacao import (
    COPILOTO_INDISPONIVEL,
    KEYWORDS_INDISPONIVEIS,
    REDACAO_INDISPONIVEL,
    registrar,
)
from .generate import (
    KeywordsUnavailable,
    analisar_ats,
    generate_cv,
    generate_cv_pipeline,
    reduzir_curriculo,
)
from .keywords import extract_keywords
from .llm import LLMUnavailable
from .rag import consultar, indexar, substituir
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    AtsAnalysis,
    GenerateCvRequest,
    GenerateCvResponse,
    GeneratePipelineResponse,
    IngestRequest,
    IngestResponse,
    KeywordsRequest,
    KeywordsResponse,
    QueryRequest,
    QueryResponse,
    ReduzirCvRequest,
    ReplaceIngestRequest,
    RedigirFormularioRequest,
    RedigirFormularioResponse,
    RedigirMensagemRequest,
    RedigirMensagemResponse,
    ScoreRequest,
    ScoreResponse,
    TaxonomiaResponse,
    TurnRequest,
    TurnResponse,
)
from .score import calcular_score

DOC_SERVICE_URL = os.getenv("DOC_SERVICE_URL", "http://localhost:8080")

logger = logging.getLogger(__name__)


def _aquecer_embeddings() -> None:
    try:
        from .rag import _model

        _model()
    except Exception as exc:
        logger.warning("aquecimento do modelo de embeddings adiado: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    prompts = carregar_prompts()
    logger.info("prompts carregados: %s", ", ".join(p.rotulo for p in prompts.values()))
    _aquecer_embeddings()
    yield


app = FastAPI(title="ai-service", lifespan=lifespan)


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
    try:
        return KeywordsResponse(keywords=extract_keywords(req.descricao))
    except LLMUnavailable as exc:
        return KeywordsResponse(
            keywords=[],
            status="PENDENTE",
            degradacao=registrar(KEYWORDS_INDISPONIVEIS, exc),
        )


@app.post("/generate-cv", response_model=GenerateCvResponse)
def generate(req: GenerateCvRequest) -> GenerateCvResponse:
    try:
        return GenerateCvResponse(markdown=generate_cv(req))
    except KeywordsUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/generate-cv-pipeline", response_model=GeneratePipelineResponse)
def generate_pipeline(req: GenerateCvRequest) -> GeneratePipelineResponse:
    try:
        return generate_cv_pipeline(req)
    except KeywordsUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/reduzir-curriculo", response_model=GeneratePipelineResponse)
def reduzir(req: ReduzirCvRequest) -> GeneratePipelineResponse:
    try:
        return reduzir_curriculo(req, req.markdown_atual)
    except KeywordsUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/analisar-ats", response_model=AtsAnalysis)
def analisar(req: GenerateCvRequest) -> AtsAnalysis:
    try:
        return analisar_ats(req)
    except KeywordsUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    return calcular_score(req.markdown, req.vaga.keywords)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    return classificar(req.titulo, req.descricao)


@app.get("/classify/taxonomy", response_model=TaxonomiaResponse)
def classify_taxonomy() -> TaxonomiaResponse:
    return TaxonomiaResponse(**taxonomia())


@app.post("/context/ingest", response_model=IngestResponse)
def context_ingest(req: IngestRequest) -> IngestResponse:
    return indexar(req.documentos)


@app.post("/context/replace", response_model=IngestResponse)
def context_replace(req: ReplaceIngestRequest) -> IngestResponse:
    if any(documento.usuario_id != req.usuario_id for documento in req.documentos):
        raise HTTPException(status_code=400, detail="replace exige documentos do usuario informado")
    return substituir(req.usuario_id, req.documentos)


@app.post("/context/query", response_model=QueryResponse)
def context_query(req: QueryRequest) -> QueryResponse:
    return consultar(req.usuario_id, req.query, req.k)


@app.post("/copiloto/turn", response_model=TurnResponse)
def copiloto_turn(req: TurnRequest) -> TurnResponse:
    try:
        return planejar_turno(req)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=registrar(COPILOTO_INDISPONIVEL, exc)) from exc


@app.post("/copiloto/redigir-mensagem", response_model=RedigirMensagemResponse)
def copiloto_redigir_mensagem(
    req: RedigirMensagemRequest,
) -> RedigirMensagemResponse:
    try:
        return redigir_mensagem(req)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=registrar(REDACAO_INDISPONIVEL, exc)) from exc


@app.post("/copiloto/redigir-formulario", response_model=RedigirFormularioResponse)
def copiloto_redigir_formulario(
    req: RedigirFormularioRequest,
) -> RedigirFormularioResponse:
    try:
        return redigir_formulario(req)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=registrar(REDACAO_INDISPONIVEL, exc)) from exc
