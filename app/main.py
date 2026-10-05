import contextvars
import json
import logging
import os
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .carregador_prompts import carregar_prompts
from .classify import classificar, taxonomia
from .copiloto import planejar_turno, planejar_turno_em_stream, redigir_formulario, redigir_mensagem
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
from .observabilidade import (
    MENSAGEM_DESLIGAMENTO,
    avisar_desligamento,
    configurar_logs,
    desligando,
    middleware_requisicao,
)
from .prontidao import prontidao
from . import telemetria
from .degradacao import Degradacao
from .llm import (
    LLMUnavailable,
    Operacao,
    OperacaoCancelada,
    PrazoEsgotado,
    Uso,
    exigir_modelo_no_boot,
    operacao,
)
from . import rag
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
    EmbeddingConsultasRequest,
    EmbeddingConsultasResponse,
    EmbeddingDocumentosRequest,
    EmbeddingDocumentosResponse,
    KeywordsRequest,
    KeywordsResponse,
    ReduzirCvRequest,
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
ESPERA_DA_FILA_S = 0.25

logger = logging.getLogger(__name__)


def _aquecer_embeddings() -> None:
    rag.configuracao()
    try:
        rag._model()
    except Exception as exc:
        logger.warning("aquecimento do modelo de embeddings adiado: %s", exc)
        return
    rag.conferir_dimensao()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configurar_logs()
    avisar_desligamento()
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
app.middleware("http")(middleware_requisicao)


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


@app.get("/ready")
def ready() -> JSONResponse:
    pronto, corpo = prontidao()
    return JSONResponse(status_code=200 if pronto else 503, content=corpo)


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
            return _com_uso(reduzir_curriculo(req), op)
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


def _embedding_indisponivel(exc: Exception) -> HTTPException:
    logger.warning("degradacao codigo=embedding_indisponivel causa=%s", exc)
    return HTTPException(status_code=503, detail="modelo de embedding indisponivel")


@app.post("/embeddings/documentos", response_model=EmbeddingDocumentosResponse)
def embeddings_documentos(req: EmbeddingDocumentosRequest) -> EmbeddingDocumentosResponse:
    modelo = rag.configuracao()
    try:
        chunks = rag.chunks_dos_documentos(req.documentos)
    except Exception as exc:
        raise _embedding_indisponivel(exc) from exc
    return EmbeddingDocumentosResponse(modelo=modelo.nome, dimensao=modelo.dimensao, chunks=chunks)


@app.post("/embeddings/consultas", response_model=EmbeddingConsultasResponse)
def embeddings_consultas(req: EmbeddingConsultasRequest) -> EmbeddingConsultasResponse:
    modelo = rag.configuracao()
    try:
        vetores = rag.vetores_de_consultas(req.consultas)
    except Exception as exc:
        raise _embedding_indisponivel(exc) from exc
    return EmbeddingConsultasResponse(
        modelo=modelo.nome, dimensao=modelo.dimensao, limiar=rag.limiar_similaridade(), vetores=vetores
    )


@app.post("/copiloto/turn", response_model=TurnResponse)
def copiloto_turn(req: TurnRequest, request: Request) -> TurnResponse:
    with _operacao_llm(request) as op:
        try:
            return _com_uso(planejar_turno(req), op)
        except LLMUnavailable as exc:
            return _indisponivel(COPILOTO_INDISPONIVEL, exc, op)


class TurnoCancelado(Exception):
    pass


def _linha(dados: dict) -> str:
    return json.dumps(dados, ensure_ascii=False) + "\n"


def _erro_do_stream(detalhe: str, op: Operacao | None, emitiu: bool) -> str:
    return _linha(
        {
            "tipo": "erro",
            "detail": detalhe,
            "interrompido": emitiu,
            "uso": _uso(op).model_dump(by_alias=True) if op else UsoLlm().model_dump(by_alias=True),
            "modelo": op.modelo if op else None,
        }
    )


def _pendentes(fila: "queue.Queue[str | None]") -> list[str]:
    linhas: list[str] = []
    while True:
        try:
            linha = fila.get_nowait()
        except queue.Empty:
            return linhas
        if linha is None:
            return linhas
        linhas.append(linha)


@app.post("/copiloto/turn/stream")
def copiloto_turn_stream(req: TurnRequest, request: Request) -> StreamingResponse:
    prazo_ms = _prazo_ms(request.headers.get(HEADER_PRAZO))
    operacao_id = request.headers.get(HEADER_OPERACAO)
    fila: queue.Queue[str | None] = queue.Queue()
    cancelado = threading.Event()

    def trabalhar() -> None:
        op: Operacao | None = None
        emitidos = 0

        def emitir(texto: str) -> None:
            nonlocal emitidos
            if cancelado.is_set():
                raise TurnoCancelado("a api encerrou a conexao do turno")
            emitidos += 1
            fila.put(_linha({"tipo": "delta", "texto": texto}))

        def avisar_uso(uso: Uso) -> None:
            fila.put(_linha({"tipo": "uso", "uso": UsoLlm(**asdict(uso)).model_dump(by_alias=True)}))

        try:
            with operacao(prazo_ms, operacao_id) as op:
                op.cancelada = cancelado
                resposta = _com_uso(planejar_turno_em_stream(req, emitir, avisar_uso), op)
                fila.put(_linha({"tipo": "fim", **resposta.model_dump(by_alias=True)}))
        except (TurnoCancelado, OperacaoCancelada) as exc:
            logger.warning("turno em stream cancelado: %s", exc)
        except LLMUnavailable as exc:
            fila.put(_erro_do_stream(registrar(COPILOTO_INDISPONIVEL, exc), op, emitidos > 0))
        except PrazoEsgotado as exc:
            logger.warning("turno em stream recusado por prazo: %s", exc)
            fila.put(_erro_do_stream("prazo da operacao esgotado", op, emitidos > 0))
        except Exception as exc:
            logger.exception("turno em stream falhou")
            fila.put(_erro_do_stream(registrar(COPILOTO_INDISPONIVEL, exc), op, emitidos > 0))
        finally:
            fila.put(None)

    threading.Thread(target=contextvars.copy_context().run, args=(trabalhar,), daemon=True).start()

    def corpo():
        emitiu = False
        try:
            while True:
                if desligando.is_set():
                    for pendente in _pendentes(fila):
                        emitiu = emitiu or '"tipo": "delta"' in pendente
                        yield pendente
                    logger.warning("turno em stream encerrado pelo desligamento do servico")
                    yield _erro_do_stream(MENSAGEM_DESLIGAMENTO, None, emitiu)
                    return
                try:
                    linha = fila.get(timeout=ESPERA_DA_FILA_S)
                except queue.Empty:
                    continue
                if linha is None:
                    return
                emitiu = emitiu or '"tipo": "delta"' in linha
                yield linha
        finally:
            cancelado.set()

    return StreamingResponse(corpo(), media_type="application/x-ndjson")


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
