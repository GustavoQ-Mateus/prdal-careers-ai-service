import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import rag
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
LIMIAR_FALSO = 0.35
VARREDURA_FALSO = (0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4)
PASSO_VARREDURA = 0.002


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


def rodar_caso(caso: dict[str, Any], falso: bool) -> dict[str, Any]:
    ferramentas = _ferramentas(caso, falso)
    chunks = _chunks(caso, ferramentas)
    consultas = caso["consultas"]
    vetores_consulta = ferramentas["consultas"]([c["termo"] for c in consultas])
    limiar = LIMIAR_FALSO if falso else rag.limiar_similaridade()
    detalhes = []
    for consulta, vetor in zip(consultas, vetores_consulta):
        notas = [(chunk, round(_cosseno(vetor, chunk["vetor"]), 4)) for chunk in chunks]
        topo = sorted(notas, key=lambda par: par[1], reverse=True)[:TOPO]
        acima = [chunk["id"] for chunk, nota in topo if nota >= limiar]
        esperados = consulta["esperados"]
        detalhes.append({
            "termo": consulta["termo"],
            "esperados": esperados,
            "topo": [{"id": chunk["id"], "similaridade": nota, "semRelacao": chunk["semRelacao"]} for chunk, nota in topo],
            "acimaDoLimiar": acima,
            "recall": round(len(set(esperados) & set(acima)) / len(esperados), 4) if esperados else None,
            "positivos": [nota for chunk, nota in notas if chunk["id"] in esperados],
            "negativos": [nota for chunk, nota in notas if chunk["semRelacao"]],
        })
    return {
        "id": caso["id"],
        "mede": caso["mede"],
        "metricas": {
            "recall_at_5": media([d["recall"] for d in detalhes]),
            "sem_relacao_acima_do_limiar": sum(1 for d in detalhes for t in d["topo"] if t["semRelacao"] and t["similaridade"] >= limiar),
            "consultas": len(detalhes),
        },
        "uso": Uso(),
        "detalhes": {"limiar": limiar, "embedding": modelo(falso), "consultas": detalhes},
    }


def _avaliar(consultas: list[dict[str, Any]], limiar: float) -> dict[str, Any]:
    recalls = []
    falsos = 0
    fora = 0
    for consulta in consultas:
        acima = {t["id"] for t in consulta["topo"] if t["similaridade"] >= limiar}
        falsos += sum(1 for t in consulta["topo"] if t["semRelacao"] and t["similaridade"] >= limiar)
        fora += len(acima - set(consulta["esperados"]))
        if consulta["esperados"]:
            recalls.append(len(set(consulta["esperados"]) & acima) / len(consulta["esperados"]))
    return {
        "limiar": limiar,
        "recall_at_5": media(recalls),
        "sem_relacao_acima_do_limiar": falsos,
        "fora_do_esperado_acima": fora,
    }


def _varredura(atual: float, menor_positivo: float, maior_negativo: float) -> list[float]:
    if atual < 0.5:
        return list(VARREDURA_FALSO)
    inicio = min(menor_positivo, maior_negativo, atual) - 0.01
    fim = max(menor_positivo, maior_negativo, atual) + 0.01
    passos = int(round((fim - inicio) / PASSO_VARREDURA))
    return [round(inicio + i * PASSO_VARREDURA, 3) for i in range(passos + 1)]


def calibracao(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    consultas = [c for r in resultados for c in r["detalhes"]["consultas"]]
    positivos = [n for c in consultas for n in c["positivos"]]
    negativos = [n for c in consultas for n in c["negativos"]]
    if not positivos or not negativos:
        return {}
    menor_positivo = min(positivos)
    maior_negativo = max(negativos)
    sugerido = round((menor_positivo + maior_negativo) / 2, 2)
    atual = resultados[0]["detalhes"]["limiar"]
    return {
        "menorPositivo": menor_positivo,
        "maiorNegativo": maior_negativo,
        "margem": round(menor_positivo - maior_negativo, 4),
        "separavel": menor_positivo > maior_negativo,
        "limiarAtual": _avaliar(consultas, atual),
        "limiarSugerido": _avaliar(consultas, sugerido),
        "varredura": [_avaliar(consultas, limiar) for limiar in _varredura(atual, menor_positivo, maior_negativo)],
        "maiorNegativoPorTermo": sorted(
            ({"termo": c["termo"], "similaridade": max(c["negativos"])} for c in consultas if c["negativos"]),
            key=lambda item: item["similaridade"],
            reverse=True,
        )[:3],
        "positivosAbaixoDoLimiarAtual": sorted(
            {f"{c['termo']} -> {t['id']} ({t['similaridade']})" for c in consultas for t in c["topo"] if t["id"] in c["esperados"] and t["similaridade"] < atual}
        ),
    }


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    consultas = [c for r in resultados for c in r["detalhes"]["consultas"]]
    return {
        "recall_at_5": media([c["recall"] for c in consultas]),
        "sem_relacao_acima_do_limiar": sum(r["metricas"]["sem_relacao_acima_do_limiar"] for r in resultados),
        "consultas": len(consultas),
    }


def extras(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    validos = [r for r in resultados if not r.get("erro")]
    return {"calibracaoDoLimiar": calibracao(validos)} if validos else {}
