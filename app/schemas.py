from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from .orcamento import (
    DESCRICAO_BULLETS,
    DESCRICAO_COMPETENCIAS,
    DESCRICAO_EXPERIENCIAS,
    DESCRICAO_FONTES,
    DESCRICAO_REPAROS,
    DESCRICAO_RESUMO,
    DESCRICAO_TITULO,
)


TipoFonte = Literal[
    "experiencia", "resumo", "skills", "formacao", "certificacao", "idiomas", "nota", "candidatura"
]


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class UsoLlm(CamelModel):
    entrada: int = 0
    saida: int = 0
    cache_lida: int = 0
    cache_escrita: int = 0
    chamadas: int = 0


class ComUso(CamelModel):
    uso: UsoLlm | None = None
    modelo: str | None = None


class Keyword(CamelModel):
    termo: str
    peso: float
    tipo: Literal["stack", "ferramenta", "metodologia", "dominio_negocio", "certificacao"] | None = None


class KeywordLlm(CamelModel):
    termo: str
    peso: float
    tipo: Literal["stack", "ferramenta", "metodologia", "dominio_negocio", "certificacao"]


class KeywordsRequest(CamelModel):
    descricao: str


class KeywordsLlmResponse(CamelModel):
    keywords: list[KeywordLlm]


class KeywordsResponse(ComUso):
    keywords: list[Keyword]
    status: Literal["VALIDAS", "PENDENTE"] = "VALIDAS"
    degradacao: str | None = None


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


class EmailContato(CamelModel):
    valor: str = ""
    principal: bool = False


class TelefoneContato(CamelModel):
    ddi: str = ""
    numero: str = ""
    principal: bool = False


class LinkContato(CamelModel):
    tipo: str = ""
    url: str = ""


class Local(CamelModel):
    pais: str = ""
    estado: str = ""
    cidade: str = ""


class ExperienciaPerfil(CamelModel):
    id: str = ""
    cargo: str = ""
    empresa: str = ""
    data_inicio_mes: int | None = None
    data_inicio_ano: int | None = None
    data_fim_mes: int | None = None
    data_fim_ano: int | None = None
    atual: bool = False
    local: Local | None = None
    local_legado: str = ""
    periodo_legado: str = ""
    descricao: str = ""
    realizacoes: list[str] = []
    texto: str = ""


class Formacao(CamelModel):
    grau: str = ""
    status: str = ""
    instituicao: str = ""
    curso: str = ""
    inicio_mes: int | None = None
    inicio_ano: int | None = None
    fim_mes: int | None = None
    fim_ano: int | None = None


class Certificacao(CamelModel):
    titulo: str = ""
    descricao: str = ""


CAMPOS_CONTATO = {"emails", "telefones", "links", "endereco"}


class PerfilMestre(CamelModel):
    nome: str = ""
    emails: list[EmailContato] = []
    telefones: list[TelefoneContato] = []
    links: list[LinkContato] = []
    endereco: Local | None = None
    resumo: str = ""
    experiencias: list[ExperienciaPerfil] = []
    formacao: list[Formacao] = []
    certificacoes: list[Certificacao] = []
    idiomas: list[str] = []
    skills: list[str] = []


class FonteContexto(CamelModel):
    id: str
    tipo: TipoFonte = "nota"
    factual: bool = False
    titulo: str = ""
    texto: str


class GenerateCvRequest(CamelModel):
    perfil_mestre: PerfilMestre
    vaga: Vaga
    keywords: list[Keyword] = []
    contexto: list[FonteContexto] = []

    @field_validator("contexto", mode="before")
    @classmethod
    def _contexto_em_texto_e_apoio(cls, valor: Any) -> Any:
        if not isinstance(valor, list):
            return valor
        return [
            {"id": f"contexto-{indice + 1}", "tipo": "nota", "factual": False, "texto": item}
            if isinstance(item, str) else item
            for indice, item in enumerate(valor)
        ]


class GenerateCvResponse(ComUso):
    markdown: str


class FraseFonte(CamelModel):
    texto: str
    fontes: list[str] = Field(description=DESCRICAO_FONTES)


class TermoFonte(CamelModel):
    termo: str
    fonte: str = Field(description="Id da fonte factual onde o termo aparece escrito")


class CategoriaCompetencias(CamelModel):
    categoria: str
    termos: list[TermoFonte]


class ExperienciaEstruturada(CamelModel):
    experiencia_id: str
    bullets: list[FraseFonte] = Field(description=DESCRICAO_BULLETS)


class FraseReparada(CamelModel):
    chave: str = Field(description="A chave da frase rejeitada, copiada sem alteracao")
    texto: str = Field(description="A frase reescrita; vazio quando nenhuma fonte citavel sustenta a frase")
    fontes: list[str] = Field(description=DESCRICAO_FONTES)


class ReescritaEstruturada(CamelModel):
    titulo: FraseFonte = Field(description=DESCRICAO_TITULO)
    resumo: list[FraseFonte] = Field(description=DESCRICAO_RESUMO)
    experiencias: list[ExperienciaEstruturada] = Field(description=DESCRICAO_EXPERIENCIAS)
    competencias: list[CategoriaCompetencias] = Field(description=DESCRICAO_COMPETENCIAS)
    reparos: list[FraseReparada] = Field(description=DESCRICAO_REPAROS)


class EstruturaCurriculo(CamelModel):
    titulo: FraseFonte | None = None
    resumo: list[FraseFonte] = []
    experiencias: list[ExperienciaEstruturada] = []
    competencias: list[CategoriaCompetencias] = []
    experiencias_omitidas: list[str] = []


class ReduzirCvRequest(GenerateCvRequest):
    estrutura: EstruturaCurriculo
    nivel: int = Field(default=1, ge=1, le=10)


class AtsAnalysis(CamelModel):
    score: int
    score_versao: int = 2
    keywords_encontradas: list[str] = []
    keywords_criticas_ausentes: list[str] = []
    pontos_eliminatorios: list[str] = []
    veredicto: str
    breakdown: dict[str, Any] = {}


class GeneratePipelineResponse(ComUso):
    markdown: str
    estrutura: EstruturaCurriculo | None = None
    analise_inicial: AtsAnalysis
    analise_final: AtsAnalysis
    degradacao: str | None = None
    prompt_version: str | None = None


class ScoreRequest(CamelModel):
    markdown: str
    vaga: Vaga


class ScoreBreakdown(CamelModel):
    keyword_match: int = Field(
        description="Percentual do peso das keywords da vaga presente no curriculo, cada termo contado uma unica vez"
    )
    densidade: int = Field(
        description="Percentual do peso das keywords da vaga sustentado na secao de experiencia; repeticao nao conta"
    )
    secoes: int = Field(
        description="Percentual das secoes obrigatorias reconhecidas por heading de nivel 2"
    )
    faltando: list[str] = []


class ScoreResponse(CamelModel):
    score: int
    score_versao: int = 2
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


class DocumentoParaEmbedding(CamelModel):
    id: str
    origem_id: str
    tipo: TipoFonte = "nota"
    texto: str


class ChunkComVetor(CamelModel):
    documento_id: str
    indice: int
    fonte_id: str
    texto: str
    vetor: list[float]


class EmbeddingDocumentosRequest(CamelModel):
    documentos: list[DocumentoParaEmbedding] = Field(max_length=200)


class EmbeddingDocumentosResponse(CamelModel):
    modelo: str
    dimensao: int
    chunks: list[ChunkComVetor]


class EmbeddingConsultasRequest(CamelModel):
    consultas: list[str] = Field(max_length=50)


class EmbeddingConsultasResponse(CamelModel):
    modelo: str
    dimensao: int
    vetores: list[list[float]]


class TrechoCandidato(CamelModel):
    id: str
    texto: str


class ConsultaComTrechos(CamelModel):
    consulta: str
    trechos: list[TrechoCandidato] = Field(max_length=50)


class FiltroTrechosRequest(CamelModel):
    consultas: list[ConsultaComTrechos] = Field(max_length=50)


class TrechosAceitos(CamelModel):
    consulta: str
    aceitos: list[str]


class FiltroTrechosResponse(CamelModel):
    consultas: list[TrechosAceitos]


class BlocoNativo(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text", "tool_use", "tool_result", "thinking", "redacted_thinking"]


class MensagemNativa(BaseModel):
    role: Literal["user", "assistant"]
    content: list[BlocoNativo]


class ToolNativa(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any]
    strict: bool | None = None


class Troca(CamelModel):
    indice: int
    mensagens: list[MensagemNativa]


class ResumoConversa(CamelModel):
    texto: str
    ate: int


class PipelineAtsContexto(CamelModel):
    oportunidade_id: str
    estado: str
    descricao: str


class TurnRequest(CamelModel):
    modo: str = "assistido"
    oportunidade_id: str | None = None
    pipeline_ats: PipelineAtsContexto | None = None
    mensagens: list[MensagemNativa] = []
    trocas: list[Troca] = []
    resumo: ResumoConversa | None = None
    tools: list[ToolNativa] = []

    def todas_as_trocas(self) -> list[Troca]:
        if self.trocas:
            return self.trocas
        return [Troca(indice=0, mensagens=self.mensagens)] if self.mensagens else []

    def todas_as_mensagens(self) -> list[MensagemNativa]:
        return [mensagem for troca in self.todas_as_trocas() for mensagem in troca.mensagens]


class TurnResponse(ComUso):
    conteudo: list[dict[str, Any]] = []
    parada: str = "end_turn"
    resumo: ResumoConversa | None = None


class ResumoLlm(CamelModel):
    resumo: str


class RedigirMensagemRequest(CamelModel):
    vaga: Vaga = Vaga()
    perfil: PerfilMestre = PerfilMestre()
    contexto: str = ""


class RedigirMensagemResponse(ComUso):
    titulo: str
    texto: str
    destino: str = ""


class MensagemLlm(CamelModel):
    titulo: str
    texto: str
    destino: str


class RespostaFormulario(CamelModel):
    campo: str
    texto: str


class RedigirFormularioRequest(CamelModel):
    vaga: Vaga = Vaga()
    perfil: PerfilMestre = PerfilMestre()
    campos: list[str] = []


class RedigirFormularioResponse(ComUso):
    titulo: str
    respostas: list[RespostaFormulario] = []
    texto: str


class FormularioLlm(CamelModel):
    titulo: str
    respostas: list[RespostaFormulario]
    texto: str
