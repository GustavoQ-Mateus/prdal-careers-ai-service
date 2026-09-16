import os
from functools import lru_cache

from .schemas import Chunk, Documento, IngestResponse, QueryResponse

MODEL_NAME = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION = "conhecimento"
CHUNK_MAX = 800


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


def _chunks(texto: str) -> list[str]:
    partes: list[str] = []
    atual = ""
    for paragrafo in texto.split("\n\n"):
        paragrafo = paragrafo.strip()
        if not paragrafo:
            continue
        if len(atual) + len(paragrafo) + 2 > CHUNK_MAX and atual:
            partes.append(atual.strip())
            atual = ""
        atual = f"{atual}\n\n{paragrafo}" if atual else paragrafo
    if atual.strip():
        partes.append(atual.strip())
    return partes


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
        partes = _chunks(doc.texto)
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
                    "titulo": doc.titulo,
                }
                for _ in partes
            ],
        )
        total += len(partes)
    return IngestResponse(indexados=total)


def consultar(usuario_id: str, query: str, k: int) -> QueryResponse:
    colecao = _collection()
    resultado = colecao.query(
        query_embeddings=_embeddings([query]),
        n_results=k,
        where={"usuarioId": usuario_id},
    )
    documentos = resultado.get("documents") or [[]]
    metadados = resultado.get("metadatas") or [[]]
    chunks = [
        Chunk(
            texto=texto,
            origem=str(meta.get("origem", "")),
            titulo=str(meta.get("titulo", "")),
        )
        for texto, meta in zip(documentos[0], metadados[0])
    ]
    return QueryResponse(chunks=chunks)
