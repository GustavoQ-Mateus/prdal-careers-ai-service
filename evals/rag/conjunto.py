import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import rag
from app.schemas import Documento
from app.text import normalize

from ..nucleo import Uso, ler_json, media

NOME = "rag"
EXIGE_CREDENCIAL = False
PASTA = Path(__file__).resolve().parent
TOPO = 5
LIMITE_FALSO = 256
PESO_CONSTANTE_FALSO = 0.3
VARREDURA = (0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4)


def modelo(falso: bool) -> str:
    return "falso" if falso else rag.MODEL_NAME


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


def _ferramentas(caso: dict[str, Any], falso: bool) -> tuple[Callable[[list[str]], list[list[float]]], Callable[[str], int], int]:
    if falso:
        return _embeddings_falsos([c["termo"] for c in caso["consultas"]]), lambda texto: len(texto.split()), LIMITE_FALSO
    if importlib.util.find_spec("sentence_transformers") is None:
        raise RuntimeError("sentence-transformers nao instalado; rode onde o modelo de embedding estiver disponivel")
    return rag._embeddings, rag.contar_tokens, rag.limite_tokens()


def _chunks(caso: dict[str, Any], contar: Callable[[str], int], limite: int) -> list[dict[str, Any]]:
    chunks = []
    for doc in caso["documentos"]:
        documento = Documento(usuario_id="avaliacao", origem=doc["tipo"], origem_id=doc["id"], tipo=doc["tipo"], texto=doc["texto"])
        partes = rag.dividir(doc["texto"], doc["tipo"], limite, contar)
        for indice, parte in enumerate(partes):
            chunks.append({
                "id": rag.id_da_fonte(documento, indice, len(partes)),
                "texto": parte,
                "semRelacao": bool(doc.get("semRelacao")),
            })
    return chunks


def _produto(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rodar_caso(caso: dict[str, Any], falso: bool) -> dict[str, Any]:
    embutir, contar, limite = _ferramentas(caso, falso)
    chunks = _chunks(caso, contar, limite)
    vetores = embutir([c["texto"] for c in chunks])
    consultas = caso["consultas"]
    vetores_consulta = embutir([rag.MOLDE_CONSULTA.format(c["termo"]) for c in consultas])
    limiar = rag.limiar_similaridade()
    detalhes = []
    for consulta, vetor in zip(consultas, vetores_consulta):
        notas = [(chunk, round(_produto(vetor, v), 4)) for chunk, v in zip(chunks, vetores)]
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
        "detalhes": {"limiar": limiar, "embedding": "falso" if falso else rag.MODEL_NAME, "consultas": detalhes},
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
        "varredura": [_avaliar(consultas, limiar) for limiar in VARREDURA],
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
