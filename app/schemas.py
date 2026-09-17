from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator
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

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalizar_keywords(cls, valor: Any) -> Any:
        if not isinstance(valor, list):
            return valor
        return [
            {"termo": item, "peso": 0.0} if isinstance(item, str) else item
            for item in valor
        ]


class ExperienciaPerfil(CamelModel):
    id: str = ""
    cargo: str = ""
    empresa: str = ""
    periodo: str = ""
    local: str = ""
    descricao: str = ""
    tecnologias: list[str] = []
    texto: str = ""


class PerfilMestre(CamelModel):
    nome: str = ""
    contato: dict[str, Any] = {}
    resumo: str = ""
    experiencias: list[ExperienciaPerfil] = []
    formacao: list[str] = []
    skills: list[str] = []


class GenerateCvRequest(CamelModel):
    perfil_mestre: PerfilMestre
    vaga: Vaga
    keywords: list[Keyword] = []
    contexto: list[str] = []


class GenerateCvResponse(CamelModel):
    markdown: str


class AtsAnalysis(CamelModel):
    score: int
    keywords_encontradas: list[str] = []
    keywords_criticas_ausentes: list[str] = []
    pontos_eliminatorios: list[str] = []
    veredicto: str
    breakdown: dict[str, Any] = {}


class GeneratePipelineResponse(CamelModel):
    markdown: str
    analise_inicial: AtsAnalysis
    analise_final: AtsAnalysis
    degradacao: str | None = None


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


class TaxonomiaResponse(CamelModel):
    categorias: list[str] = []
    niveis: list[str] = []


class Documento(CamelModel):
    usuario_id: str
    origem: str
    origem_id: str
    titulo: str = ""
    texto: str


class IngestRequest(CamelModel):
    documentos: list[Documento]


class IngestResponse(CamelModel):
    indexados: int


class QueryRequest(CamelModel):
    usuario_id: str
    query: str
    k: int = 5


class Chunk(CamelModel):
    texto: str
    origem: str
    titulo: str


class QueryResponse(CamelModel):
    chunks: list[Chunk]


class MensagemTurno(CamelModel):
    papel: str
    conteudo: str
    tool: str | None = None


class ToolSpec(CamelModel):
    nome: str
    efeito: str
    descricao: str = ""
    parametros: dict[str, Any] = {}


class TurnRequest(CamelModel):
    modo: str = "assistido"
    oportunidade_id: str | None = None
    mensagens: list[MensagemTurno] = []
    tools: list[ToolSpec] = []


class TurnResponse(CamelModel):
    tipo: str
    texto: str | None = None
    tool: str | None = None
    args: dict[str, Any] = {}


class RedigirMensagemRequest(CamelModel):
    vaga: Vaga = Vaga()
    perfil: PerfilMestre = PerfilMestre()
    contexto: str = ""


class RedigirMensagemResponse(CamelModel):
    titulo: str
    texto: str
    destino: str = ""


class RespostaFormulario(CamelModel):
    campo: str
    texto: str


class RedigirFormularioRequest(CamelModel):
    vaga: Vaga = Vaga()
    perfil: PerfilMestre = PerfilMestre()
    campos: list[str] = []


class RedigirFormularioResponse(CamelModel):
    titulo: str
    respostas: list[RespostaFormulario] = []
    texto: str
