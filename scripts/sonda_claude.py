import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm
from app.copiloto import planejar_turno, redigir_formulario, redigir_mensagem
from app.generate import generate_cv_pipeline
from app.keywords import extract_keywords
from app.llm import LLMUnavailable, operacao
from app.schemas import (
    GenerateCvRequest,
    MensagemTurno,
    RedigirFormularioRequest,
    RedigirMensagemRequest,
    ToolSpec,
    TurnRequest,
)

RODADAS = 2

VAGA = {
    "titulo": "Desenvolvedor Backend Pleno",
    "empresa": "Empresa Exemplo",
    "descricao": (
        "Buscamos pessoa desenvolvedora backend para APIs REST em Python com FastAPI, "
        "PostgreSQL e Docker. Desejavel experiencia com AWS, filas SQS e testes automatizados."
    ),
}

PERFIL = {
    "nome": "Pessoa Exemplo",
    "resumo": "Desenvolvedora backend com APIs REST em Python.",
    "experiencias": [
        {
            "cargo": "Desenvolvedora Backend",
            "empresa": "Companhia Ficticia",
            "dataInicioMes": 1,
            "dataInicioAno": 2022,
            "atual": True,
            "descricao": "Tecnologias: Python, FastAPI, PostgreSQL, Docker",
            "realizacoes": [
                "Desenvolvi APIs REST em Python com FastAPI para o modulo de pedidos.",
                "Participei da migracao do banco para PostgreSQL com o time de dados.",
            ],
        }
    ],
    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Testes automatizados"],
}

KEYWORDS = [
    {"termo": "Python", "peso": 1.0},
    {"termo": "FastAPI", "peso": 0.9},
    {"termo": "PostgreSQL", "peso": 0.8},
    {"termo": "Docker", "peso": 0.7},
    {"termo": "AWS", "peso": 0.5},
]


class _Mensagens:
    def __init__(self, gravador: "Gravador", alvo: Any):
        self._gravador = gravador
        self._alvo = alvo

    def create(self, **requisicao: Any) -> Any:
        inicio = time.perf_counter()
        resposta = self._alvo.messages.create(**requisicao)
        self._gravador.chamadas.append((resposta, (time.perf_counter() - inicio) * 1000))
        return resposta


class _ComOpcoes:
    def __init__(self, gravador: "Gravador", alvo: Any):
        self.messages = _Mensagens(gravador, alvo)


class Gravador:
    def __init__(self, real: Any):
        self._real = real
        self.chamadas: list[tuple[Any, float]] = []

    def with_options(self, **opcoes: Any) -> _ComOpcoes:
        return _ComOpcoes(self, self._real.with_options(**opcoes))


def _casos() -> list[tuple[str, Callable[[], Any]]]:
    turno = TurnRequest(
        oportunidade_id="op-exemplo",
        mensagens=[MensagemTurno(papel="user", conteudo="Quais sao os proximos passos para esta vaga?")],
        tools=[
            ToolSpec(nome="buscar_oportunidade", efeito="leitura", descricao="Le a oportunidade em foco"),
            ToolSpec(nome="analisar_ats", efeito="leitura", descricao="Analisa a aderencia do perfil a vaga"),
        ],
    )
    geracao = GenerateCvRequest.model_validate(
        {"perfilMestre": PERFIL, "vaga": {**VAGA, "keywords": KEYWORDS}, "keywords": KEYWORDS, "contexto": []}
    )
    return [
        ("keywords", lambda: extract_keywords(VAGA["descricao"])),
        ("copiloto_turno", lambda: planejar_turno(turno)),
        (
            "redigir_mensagem",
            lambda: redigir_mensagem(RedigirMensagemRequest.model_validate({"vaga": VAGA, "perfil": PERFIL})),
        ),
        (
            "redigir_formulario",
            lambda: redigir_formulario(
                RedigirFormularioRequest.model_validate(
                    {"vaga": VAGA, "perfil": PERFIL, "campos": ["Por que esta vaga?", "Pretensao salarial"]}
                )
            ),
        ),
        ("reescrita", lambda: generate_cv_pipeline(geracao)),
    ]


def _imprimir(chamador: str, rodada: int, validou: str, chamadas: list[tuple[Any, float]]) -> None:
    for indice, (resposta, latencia_ms) in enumerate(chamadas, start=1):
        usage = resposta.usage
        print(
            f"{chamador:<20} rodada={rodada} chamada={indice} modelo={resposta.model} "
            f"stop={resposta.stop_reason} entrada={usage.input_tokens} saida={usage.output_tokens} "
            f"cache_escrita={usage.cache_creation_input_tokens or 0} "
            f"cache_lida={usage.cache_read_input_tokens or 0} latencia_ms={latencia_ms:.0f} validou={validou}"
        )
    if not chamadas:
        print(f"{chamador:<20} rodada={rodada} sem chamada ao modelo validou={validou}")


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY ausente: defina a chave da Anthropic para rodar a sonda ao vivo.", file=sys.stderr)
        return 2
    if not llm.modelo_configurado():
        print("AI_MODEL ausente: defina o modelo, por exemplo AI_MODEL=claude-sonnet-5.", file=sys.stderr)
        return 2
    gravador = Gravador(llm.cliente())
    llm.definir_cliente(gravador)
    falhas = 0
    for chamador, executar in _casos():
        for rodada in range(1, RODADAS + 1):
            gravador.chamadas.clear()
            try:
                with operacao(operacao_id=f"sonda:{chamador}:{rodada}"):
                    resultado = executar()
                degradacao = getattr(resultado, "degradacao", None)
                validou = "nao (" + degradacao + ")" if degradacao else "sim"
            except LLMUnavailable as exc:
                validou = f"nao ({exc})"
            if validou != "sim":
                falhas += 1
            _imprimir(chamador, rodada, validou, list(gravador.chamadas))
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
