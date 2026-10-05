import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from .casamento import termo_presente
from .schemas import ChunkComVetor, DocumentoParaEmbedding, TrechoCandidato

TIPOS_EM_PARAGRAFOS = ("nota", "candidatura")
TOKENS_ESPECIAIS = 2
DIMENSAO_PADRAO = 384

logger = logging.getLogger(__name__)

_FRASE_RE = re.compile(r"(?<=[.!?;])\s+")


@dataclass(frozen=True)
class ModeloEmbedding:
    nome: str
    dimensao: int
    prefixo_consulta: str
    prefixo_documento: str
    molde_consulta: str


MODELOS = {
    modelo.nome: modelo
    for modelo in (
        ModeloEmbedding(
            nome="intfloat/multilingual-e5-small",
            dimensao=384,
            prefixo_consulta="query: ",
            prefixo_documento="passage: ",
            molde_consulta="experiência com {}",
        ),
        ModeloEmbedding(
            nome="paraphrase-multilingual-MiniLM-L12-v2",
            dimensao=384,
            prefixo_consulta="",
            prefixo_documento="",
            molde_consulta="experiencia com {}",
        ),
    )
}
MODELO_PADRAO = "intfloat/multilingual-e5-small"


class ModeloDesconhecido(RuntimeError):
    pass


class DimensaoIncompativel(RuntimeError):
    pass


def configuracao() -> ModeloEmbedding:
    nome = os.getenv("EMBED_MODEL", "").strip() or MODELO_PADRAO
    modelo = MODELOS.get(nome)
    if modelo is None:
        raise ModeloDesconhecido(f"EMBED_MODEL {nome} nao tem prefixos calibrados; use um de {sorted(MODELOS)}")
    return modelo


def dimensao_esperada() -> int:
    try:
        return int(os.getenv("EMBED_DIMENSAO", str(DIMENSAO_PADRAO)))
    except ValueError:
        return DIMENSAO_PADRAO


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        configuracao().nome,
        device="cpu",
        model_kwargs={"low_cpu_mem_usage": False},
    )


def conferir_dimensao() -> None:
    modelo = configuracao()
    esperada = dimensao_esperada()
    carregado = _model()
    medir = getattr(carregado, "get_embedding_dimension", None) or carregado.get_sentence_embedding_dimension
    carregada = int(medir())
    if modelo.dimensao != esperada or carregada != esperada:
        raise DimensaoIncompativel(
            f"o modelo {modelo.nome} gera vetores de {carregada} dimensoes e o banco espera {esperada}; "
            "trocar de modelo exige migrar a coluna e reindexar"
        )


def limite_tokens() -> int:
    return int(_model().max_seq_length)


def contar_tokens(texto: str) -> int:
    return len(_model().tokenizer.tokenize(configuracao().prefixo_documento + texto)) + TOKENS_ESPECIAIS


def _pedacos(texto: str, limite: int, contar: Callable[[str], int]) -> list[str]:
    if contar(texto) <= limite:
        return [texto]
    frases = _FRASE_RE.split(texto)
    unidades = frases if len(frases) > 1 else texto.split()
    if len(unidades) <= 1:
        return [texto]
    partes: list[str] = []
    atual = ""
    for unidade in unidades:
        candidato = f"{atual} {unidade}".strip()
        if atual and contar(candidato) > limite:
            partes.append(atual)
            atual = unidade
        else:
            atual = candidato
    if atual:
        partes.append(atual)
    if len(partes) == 1:
        return partes
    return [pedaco for parte in partes for pedaco in _pedacos(parte, limite, contar)]


def dividir(
    texto: str,
    tipo: str,
    limite: int | None = None,
    contar: Callable[[str], int] | None = None,
) -> list[str]:
    limpo = texto.strip()
    if not limpo:
        return []
    if tipo not in TIPOS_EM_PARAGRAFOS:
        return [limpo]
    teto = limite if limite is not None else limite_tokens()
    medir = contar or contar_tokens
    paragrafos = [p.strip() for p in re.split(r"\n\s*\n", limpo) if p.strip()]
    return [pedaco for paragrafo in paragrafos for pedaco in _pedacos(paragrafo, teto, medir)]


def id_da_fonte(origem_id: str, tipo: str, indice: int, total: int) -> str:
    if tipo not in TIPOS_EM_PARAGRAFOS:
        return origem_id
    return f"{origem_id}#{indice + 1}" if total > 1 else origem_id


def _vetores(textos: list[str]) -> list[list[float]]:
    if not textos:
        return []
    return _model().encode(textos, normalize_embeddings=True).tolist()


def vetores_de_documentos(textos: list[str]) -> list[list[float]]:
    prefixo = configuracao().prefixo_documento
    return _vetores([prefixo + texto for texto in textos])


def vetores_de_consultas(consultas: list[str]) -> list[list[float]]:
    modelo = configuracao()
    return _vetores([modelo.prefixo_consulta + modelo.molde_consulta.format(consulta.strip()) for consulta in consultas])


def chunks_dos_documentos(
    documentos: list[DocumentoParaEmbedding],
    limite: int | None = None,
    contar: Callable[[str], int] | None = None,
    embutir: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[ChunkComVetor]:
    partes: list[tuple[DocumentoParaEmbedding, int, str, str]] = []
    for doc in documentos:
        pedacos = dividir(doc.texto, doc.tipo, limite, contar)
        for indice, pedaco in enumerate(pedacos):
            partes.append((doc, indice, id_da_fonte(doc.origem_id, doc.tipo, indice, len(pedacos)), pedaco))
    vetores = (embutir or vetores_de_documentos)([texto for _, _, _, texto in partes])
    return [
        ChunkComVetor(documento_id=doc.id, indice=indice, fonte_id=fonte_id, texto=texto, vetor=vetor)
        for (doc, indice, fonte_id, texto), vetor in zip(partes, vetores)
    ]


def trechos_que_casam(consulta: str, trechos: list[TrechoCandidato]) -> list[str]:
    return [trecho.id for trecho in trechos if termo_presente(consulta, trecho.texto)]
