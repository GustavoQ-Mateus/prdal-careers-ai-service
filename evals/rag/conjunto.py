import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import rag
from app.casamento import termo_presente
from app.schemas import DocumentoParaEmbedding
from app.text import normalize

from ..nucleo import Uso, ler_json, media

NOME = "rag"
EXIGE_CREDENCIAL = False
PASTA = Path(__file__).resolve().parent
TOPO = 5
LIMITE_FALSO = 256
PESO_CONSTANTE_FALSO = 0.3
MOLDE_FALSO = "experiencia com {}"
ANTIGOS = ("corpus_variado", "nada_passa")


def modelo(falso: bool) -> str:
    return "falso" if falso else rag.configuracao().nome


def carregar() -> list[dict[str, Any]]:
    return ler_json(PASTA / "casos.json")


def _embeddings_falsos(vocabulario: list[str]) -> Callable[[list[str]], list[list[float]]]:
    termos = [normalize(termo) for termo in vocabulario]

    def embutir(textos: list[str]) -> list[list[float]]:
        vetores = []
        for texto in textos:
            normalizado = normalize(texto)
            vetor = [1.0 if termo in normalizado else 0.0 for termo in termos] + [PESO_CONSTANTE_FALSO]
            norma = math.sqrt(sum(v * v for v in vetor))
            vetores.append([v / norma for v in vetor])
        return vetores

    return embutir


def _ferramentas(caso: dict[str, Any], falso: bool) -> dict[str, Any]:
    if falso:
        falsos = _embeddings_falsos([c["termo"] for c in caso["consultas"]])
        return {
            "documentos": falsos,
            "consultas": lambda termos: falsos([MOLDE_FALSO.format(t) for t in termos]),
            "contar": lambda texto: len(texto.split()),
            "limite": LIMITE_FALSO,
        }
    if importlib.util.find_spec("sentence_transformers") is None:
        raise RuntimeError("sentence-transformers nao instalado; rode onde o modelo de embedding estiver disponivel")
    return {"documentos": None, "consultas": rag.vetores_de_consultas, "contar": None, "limite": None}


def _chunks(caso: dict[str, Any], ferramentas: dict[str, Any]) -> list[dict[str, Any]]:
    documentos = [DocumentoParaEmbedding(id=doc["id"], origem_id=doc["id"], tipo=doc["tipo"], texto=doc["texto"]) for doc in caso["documentos"]]
    sem_relacao = {doc["id"] for doc in caso["documentos"] if doc.get("semRelacao")}
    chunks = rag.chunks_dos_documentos(documentos, ferramentas["limite"], ferramentas["contar"], ferramentas["documentos"])
    return [{"id": c.fonte_id, "texto": c.texto, "vetor": c.vetor, "semRelacao": c.documento_id in sem_relacao} for c in chunks]


def _cosseno(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def no_contexto(termo: str, topo: list[dict[str, Any]]) -> list[str]:
    return [chunk["id"] for chunk in topo if termo_presente(termo, chunk["texto"])]


def rodar_caso(caso: dict[str, Any], falso: bool) -> dict[str, Any]:
    ferramentas = _ferramentas(caso, falso)
    chunks = _chunks(caso, ferramentas)
    consultas = caso["consultas"]
    vetores_consulta = ferramentas["consultas"]([c["termo"] for c in consultas])
    detalhes = []
    for consulta, vetor in zip(consultas, vetores_consulta):
        notas = [(chunk, round(_cosseno(vetor, chunk["vetor"]), 4)) for chunk in chunks]
        topo = [chunk for chunk, _ in sorted(notas, key=lambda par: par[1], reverse=True)[:TOPO]]
        similaridade = {chunk["id"]: nota for chunk, nota in notas}
        contexto = no_contexto(consulta["termo"], topo)
        esperados = consulta["esperados"]
        detalhes.append({
            "termo": consulta["termo"],
            "esperados": esperados,
            "topo": [{"id": c["id"], "similaridade": similaridade[c["id"]], "semRelacao": c["semRelacao"], "noContexto": c["id"] in contexto} for c in topo],
            "contexto": contexto,
            "recall": round(len(set(esperados) & set(contexto)) / len(esperados), 4) if esperados else None,
        })
    return {
        "id": caso["id"],
        "mede": caso["mede"],
        "metricas": {
            "recall_at_5": media([d["recall"] for d in detalhes]),
            "sem_relacao_no_contexto": sum(1 for d in detalhes for t in d["topo"] if t["semRelacao"] and t["noContexto"]),
            "consultas": len(detalhes),
        },
        "uso": Uso(),
        "detalhes": {"filtro": "casamento de termos", "embedding": modelo(falso), "consultas": detalhes},
    }


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    consultas = [c for r in resultados for c in r["detalhes"]["consultas"]]
    antigas = [c for r in resultados if r["id"] in ANTIGOS for c in r["detalhes"]["consultas"]]
    return {
        "recall_at_5": media([c["recall"] for c in consultas]),
        "recall_at_5_antigas": media([c["recall"] for c in antigas]),
        "sem_relacao_no_contexto": sum(r["metricas"]["sem_relacao_no_contexto"] for r in resultados),
        "consultas": len(consultas),
    }


def extras(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    consultas = [c for r in resultados if not r.get("erro") for c in r["detalhes"]["consultas"]]
    return {
        "esperadosForaDoContexto": sorted(
            f"{c['termo']} -> {t['id']} ({t['similaridade']})"
            for c in consultas for t in c["topo"] if t["id"] in c["esperados"] and not t["noContexto"]
        ),
        "esperadosForaDoTopo": sorted(
            f"{c['termo']} -> {e}" for c in consultas for e in c["esperados"] if e not in {t["id"] for t in c["topo"]}
        ),
    }
