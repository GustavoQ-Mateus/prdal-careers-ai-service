import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.copiloto import planejar_turno
from app.generate import generate_cv_pipeline
from app.llm import LLMUnavailable
from app.schemas import (
    GenerateCvRequest,
    Keyword,
    PerfilMestre,
    ToolSpec,
    TurnRequest,
    Vaga,
)

MODELOS = ("llama-3.3-70b-versatile", "openai/gpt-oss-120b")
RODADAS = max(1, int(os.getenv("EVAL_RUNS", "1")))

TOOLS = [
    ToolSpec(
        nome="registrar_oportunidade",
        efeito="escrita",
        descricao="Registra uma oportunidade",
        parametros={"titulo": "texto", "empresa": "texto", "descricao": "texto"},
    ),
    ToolSpec(
        nome="gerar_curriculo",
        efeito="escrita",
        descricao="Inicia uma geracao",
        parametros={"oportunidadeId": "id"},
    ),
]

TURNO = TurnRequest(
    modo="assistido",
    mensagens=[
        {
            "papel": "user",
            "conteudo": "Colei uma vaga de Pessoa Engenheira de Software na Acme. Registre e prepare o curriculo.",
        }
    ],
    tools=TOOLS,
)

GERACAO = GenerateCvRequest(
    perfil_mestre=PerfilMestre(
        nome="Candidato de avaliacao",
        resumo="Engenheiro de software com experiencia em APIs e produtos web.",
        experiencias=[],
        skills=["Python", "TypeScript", "PostgreSQL"],
    ),
    vaga=Vaga(
        titulo="Pessoa Engenheira de Software",
        empresa="Acme",
        descricao="Buscamos experiencia em APIs, Python, TypeScript e PostgreSQL.",
        keywords=[
            Keyword(termo="Python", peso=1),
            Keyword(termo="TypeScript", peso=1),
            Keyword(termo="PostgreSQL", peso=1),
        ],
    ),
)


def avaliar(modelo: str) -> dict[str, object]:
    os.environ["AI_MODEL"] = modelo
    turnos: list[dict[str, object]] = []
    geracoes: list[dict[str, object]] = []

    for _ in range(RODADAS):
        inicio = time.perf_counter()
        try:
            resultado = planejar_turno(TURNO)
            turnos.append(
                {
                    "valido": resultado.tipo in {"texto", "tool_call"},
                    "tipo": resultado.tipo,
                    "tool": resultado.tool,
                    "latencia_ms": round((time.perf_counter() - inicio) * 1000),
                }
            )
        except LLMUnavailable as exc:
            turnos.append(
                {
                    "valido": False,
                    "erro": type(exc).__name__,
                    "mensagem": str(exc),
                    "latencia_ms": round((time.perf_counter() - inicio) * 1000),
                }
            )

        inicio = time.perf_counter()
        try:
            resultado = generate_cv_pipeline(GERACAO)
            geracoes.append(
                {
                    "valido": bool(resultado.markdown.strip()),
                    "score_inicial": resultado.analise_inicial.score,
                    "score_final": resultado.analise_final.score,
                    "latencia_ms": round((time.perf_counter() - inicio) * 1000),
                }
            )
        except LLMUnavailable as exc:
            geracoes.append(
                {
                    "valido": False,
                    "erro": type(exc).__name__,
                    "mensagem": str(exc),
                    "latencia_ms": round((time.perf_counter() - inicio) * 1000),
                }
            )

    return {"modelo": modelo, "rodadas": RODADAS, "turnos": turnos, "geracoes": geracoes}


print(json.dumps({"modelos": [avaliar(modelo) for modelo in MODELOS]}, ensure_ascii=False, indent=2))

