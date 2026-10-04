import logging
import os
import re
from collections.abc import Callable
from functools import lru_cache
from typing import get_args

from .schemas import Chunk, Documento, IngestResponse, QueryResponse, TipoFonte

MODEL_NAME = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION = "conhecimento"
TIPOS_EM_PARAGRAFOS = ("nota", "candidatura")
TETO_CONSULTAS = 12
TETO_CHUNKS = 20
LIMIAR_PADRAO = 0.35
MOLDE_CONSULTA = "experiencia com {}"
TOKENS_ESPECIAIS = 2

logger = logging.getLogger(__name__)

_FRASE_RE = re.compile(r"(?<=[.!?;])\s+")


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"low_cpu_mem_usage": False},
    )


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(COLLECTION)


def limite_tokens() -> int:
    return int(_model().max_seq_length)


def contar_tokens(texto: str) -> int:
    return len(_model().tokenizer.tokenize(texto)) + TOKENS_ESPECIAIS


def limiar_similaridade() -> float:
    try:
        valor = float(os.getenv("RAG_LIMIAR_SIMILARIDADE", str(LIMIAR_PADRAO)))
    except ValueError:
        return LIMIAR_PADRAO
    return valor if -1.0 <= valor <= 1.0 else LIMIAR_PADRAO


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


def id_da_fonte(doc: Documento, indice: int, total: int) -> str:
    if doc.tipo not in TIPOS_EM_PARAGRAFOS:
        return doc.origem_id
    return f"{doc.origem_id}#{indice + 1}" if total > 1 else doc.origem_id


def _embeddings(textos: list[str]) -> list[list[float]]:
    return _model().encode(textos, normalize_embeddings=True).tolist()


def indexar(documentos: list[Documento]) -> IngestResponse:
    colecao = _collection()
    total = 0
    for doc in documentos:
        colecao.delete(
            where={
                "$and": [
                    {"usuarioId": doc.usuario_id},
                    {"origem": doc.origem},
                    {"origemId": doc.origem_id},
                ]
            }
        )
        partes = dividir(doc.texto, doc.tipo)
        if not partes:
            continue
        colecao.add(
            ids=[f"{doc.usuario_id}:{doc.origem}:{doc.origem_id}:{i}" for i in range(len(partes))],
            documents=partes,
            embeddings=_embeddings(partes),
            metadatas=[
                {
                    "usuarioId": doc.usuario_id,
                    "origem": doc.origem,
                    "origemId": doc.origem_id,
                    "fonteId": id_da_fonte(doc, i, len(partes)),
                    "tipo": doc.tipo,
                    "factual": doc.factual,
                    "titulo": doc.titulo,
                }
                for i in range(len(partes))
            ],
        )
        total += len(partes)
    return IngestResponse(indexados=total)


def substituir(usuario_id: str, documentos: list[Documento]) -> IngestResponse:
    colecao = _collection()
    colecao.delete(where={"usuarioId": usuario_id})
    return indexar(documentos)


def similaridade(distancia: float) -> float:
    return 1.0 - distancia / 2.0


def consultas_unicas(consultas: list[str]) -> list[str]:
    vistas: set[str] = set()
    unicas: list[str] = []
    for consulta in consultas:
        limpa = consulta.strip()
        chave = limpa.lower()
        if limpa and chave not in vistas:
            vistas.add(chave)
            unicas.append(limpa)
    return unicas[:TETO_CONSULTAS]


def _tipo_valido(valor: object) -> str:
    tipo = str(valor or "")
    return tipo if tipo in get_args(TipoFonte) else "nota"


def consultar(usuario_id: str, consultas: list[str], k: int) -> QueryResponse:
    unicas = consultas_unicas(consultas)
    if not unicas:
        return QueryResponse(chunks=[])
    colecao = _collection()
    resultado = colecao.query(
        query_embeddings=_embeddings([MOLDE_CONSULTA.format(consulta) for consulta in unicas]),
        n_results=k,
        where={"usuarioId": usuario_id},
        include=["documents", "metadatas", "distances"],
    )
    limiar = limiar_similaridade()
    melhores: dict[str, tuple[float, Chunk]] = {}
    for documentos, metadados, distancias in zip(
        resultado.get("documents") or [],
        resultado.get("metadatas") or [],
        resultado.get("distances") or [],
    ):
        for texto, meta, distancia in zip(documentos, metadados, distancias):
            nota = similaridade(float(distancia))
            if nota < limiar:
                continue
            fonte_id = str(meta.get("fonteId") or meta.get("origemId") or "")
            if not fonte_id:
                continue
            chunk = Chunk(
                id=fonte_id,
                tipo=_tipo_valido(meta.get("tipo")),
                factual=meta.get("factual") is True,
                titulo=str(meta.get("titulo", "")),
                texto=texto,
                origem=str(meta.get("origem", "")),
                similaridade=round(nota, 4),
            )
            if fonte_id not in melhores or nota > melhores[fonte_id][0]:
                melhores[fonte_id] = (nota, chunk)
    ordenados = sorted(melhores.values(), key=lambda par: par[0], reverse=True)
    logger.info(
        "rag consultas=%s candidatos=%s acima_do_limiar=%s limiar=%s",
        len(unicas), sum(len(d) for d in resultado.get("documents") or []), len(ordenados), limiar,
    )
    return QueryResponse(chunks=[chunk for _, chunk in ordenados[:TETO_CHUNKS]])
