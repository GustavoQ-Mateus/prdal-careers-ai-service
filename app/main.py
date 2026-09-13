import os

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from .classify import classificar
from .generate import generate_cv
from .keywords import extract_keywords
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    GenerateCvRequest,
    GenerateCvResponse,
    KeywordsRequest,
    KeywordsResponse,
    ScoreRequest,
    ScoreResponse,
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
