import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
from . import telemetria
from .degradacao import Degradacao
from .llm import LLMUnavailable, Operacao, PrazoEsgotado, exigir_modelo_no_boot, operacao
from .rag import consultar, indexar, substituir
from .seguranca import (
    HEADER_SERVICO,
    exigir_servico,
    exigir_token_no_boot,
    rotas_de_documentacao,
    token_servico,
)
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
    UsoLlm,
)
from .score import calcular_score

DOC_SERVICE_URL = os.getenv("DOC_SERVICE_URL", "http://localhost:8080")
HEADER_PRAZO = "X-Prdal-Prazo-Ms"
HEADER_OPERACAO = "X-Prdal-Operacao"

logger = logging.getLogger(__name__)


def _aquecer_embeddings() -> None:
    try:
        from .rag import _model

        _model()
    except Exception as exc:
        logger.warning("aquecimento do modelo de embeddings adiado: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    exigir_token_no_boot()
    exigir_modelo_no_boot()
    telemetria.configurar()
    prompts = carregar_prompts()
    logger.info("prompts carregados: %s", ", ".join(p.rotulo for p in prompts.values()))
    _aquecer_embeddings()
    yield
    telemetria.encerrar()


app = FastAPI(title="ai-service", lifespan=lifespan, **rotas_de_documentacao())
app.middleware("http")(exigir_servico)


@app.exception_handler(PrazoEsgotado)
def prazo_esgotado(_request: Request, exc: PrazoEsgotado) -> JSONResponse:
    logger.warning("operacao recusada por prazo esgotado: %s", exc)
    return JSONResponse(status_code=504, content={"detail": "prazo da operacao esgotado"})


def _prazo_ms(valor: str | None) -> int | None:
    if valor is None or not valor.strip():
        return None
    try:
        return int(float(valor))
    except ValueError:
        return None


def _operacao_llm(request: Request):
    return operacao(_prazo_ms(request.headers.get(HEADER_PRAZO)), request.headers.get(HEADER_OPERACAO))


def _uso(op: Operacao) -> UsoLlm:
    return UsoLlm(**asdict(op.uso))


def _com_uso(resposta, op: Operacao):
    resposta.uso = _uso(op)
    resposta.modelo = op.modelo
    return resposta


def _indisponivel(degradacao: Degradacao, exc: Exception, op: Operacao) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": registrar(degradacao, exc),
            "uso": _uso(op).model_dump(by_alias=True),
            "modelo": op.modelo,
        },
    )


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
        resp = await client.get(
            f"{DOC_SERVICE_URL}/hello",
            headers={HEADER_SERVICO: token_servico()},
        )
        resp.raise_for_status()
        downstream = HelloResponse.model_validate(resp.json())
    return HelloResponse(
        service="ai-service",
        message=hop.message,
        chain=[hop, *downstream.chain],
    )


@app.post("/keywords", response_model=KeywordsResponse)
def keywords(req: KeywordsRequest, request: Request) -> KeywordsResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(KeywordsResponse(keywords=extract_keywords(req.descricao)), op)
        except LLMUnavailable as exc:
            return _com_uso(
                KeywordsResponse(
                    keywords=[],
                    status="PENDENTE",
                    degradacao=registrar(KEYWORDS_INDISPONIVEIS, exc),
                ),
                op,
            )


@app.post("/generate-cv", response_model=GenerateCvResponse)
def generate(req: GenerateCvRequest, request: Request) -> GenerateCvResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(GenerateCvResponse(markdown=generate_cv(req)), op)
        except KeywordsUnavailable as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/generate-cv-pipeline", response_model=GeneratePipelineResponse)
def generate_pipeline(req: GenerateCvRequest, request: Request) -> GeneratePipelineResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(generate_cv_pipeline(req), op)
        except KeywordsUnavailable as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/reduzir-curriculo", response_model=GeneratePipelineResponse)
def reduzir(req: ReduzirCvRequest, request: Request) -> GeneratePipelineResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(reduzir_curriculo(req, req.markdown_atual), op)
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
def copiloto_turn(req: TurnRequest, request: Request) -> TurnResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(planejar_turno(req), op)
        except LLMUnavailable as exc:
            return _indisponivel(COPILOTO_INDISPONIVEL, exc, op)


@app.post("/copiloto/redigir-mensagem", response_model=RedigirMensagemResponse)
def copiloto_redigir_mensagem(
    req: RedigirMensagemRequest, request: Request
) -> RedigirMensagemResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(redigir_mensagem(req), op)
        except LLMUnavailable as exc:
            return _indisponivel(REDACAO_INDISPONIVEL, exc, op)


@app.post("/copiloto/redigir-formulario", response_model=RedigirFormularioResponse)
def copiloto_redigir_formulario(
    req: RedigirFormularioRequest, request: Request
) -> RedigirFormularioResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(redigir_formulario(req), op)
        except LLMUnavailable as exc:
            return _indisponivel(REDACAO_INDISPONIVEL, exc, op)
