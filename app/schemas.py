from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Keyword(CamelModel):
    termo: str
    peso: float


class KeywordsRequest(CamelModel):
    descricao: str


class KeywordsResponse(CamelModel):
    keywords: list[Keyword]


class Vaga(CamelModel):
    titulo: str = ""
    empresa: str = ""
    descricao: str = ""
    keywords: list[Keyword] = []


class PerfilMestre(CamelModel):
    nome: str = ""
    contato: dict[str, Any] = {}
    resumo: str = ""
    experiencias: list[Any] = []
    formacao: list[Any] = []
    skills: list[Any] = []


class GenerateCvRequest(CamelModel):
    perfil_mestre: PerfilMestre
    vaga: Vaga
    keywords: list[Keyword] = []
    contexto: list[str] = []


class GenerateCvResponse(CamelModel):
    markdown: str


class ScoreRequest(CamelModel):
    markdown: str
    vaga: Vaga


class ScoreBreakdown(CamelModel):
    keyword_match: int
    densidade: int
    secoes: int
    faltando: list[str] = []


class ScoreResponse(CamelModel):
    score: int
    breakdown: ScoreBreakdown


class ClassifyRequest(CamelModel):
    titulo: str = ""
    descricao: str = ""


class ClassifyResponse(CamelModel):
    categoria: str
    nivel: str
