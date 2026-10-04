import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm
from app.contexto import contador
from app.copiloto import SYSTEM_TURNO, planejar_turno, redigir_formulario, redigir_mensagem
from app.generate import generate_cv_pipeline
from app.keywords import extract_keywords
from app.llm import LLMUnavailable, operacao
from app.schemas import (
    GenerateCvRequest,
    RedigirFormularioRequest,
    RedigirMensagemRequest,
    TurnRequest,
)

RODADAS = 2
CHAMADOR_GERACAO = "reescrita"
CHAMADOR_NATIVO = "turno_nativo"
PEDIDO_NATIVO = "Abra os detalhes da oportunidade em foco e me diga o titulo e a empresa."
OPORTUNIDADE_NATIVA = {
    "id": "op-exemplo",
    "titulo": "Desenvolvedor Backend Pleno",
    "empresa": "Empresa Exemplo",
    "descricao": "APIs REST em Python com FastAPI e PostgreSQL.",
}

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

PERFIL_GERACAO = {
    "nome": "Pessoa Exemplo",
    "resumo": "Desenvolvedora backend com APIs REST em Python e Java.",
    "experiencias": [
        {
            "id": "atual",
            "cargo": "Desenvolvedora Backend",
            "empresa": "Companhia Ficticia",
            "dataInicioMes": 1,
            "dataInicioAno": 2023,
            "atual": True,
            "descricao": (
                "- Desenvolvi APIs REST em Python com FastAPI para o modulo de pedidos.\n"
                "- Participei da migracao do banco para PostgreSQL com o time de dados.\n"
                "- Reduzi o tempo de resposta das consultas em 30% com indices no PostgreSQL."
            ),
        },
        {
            "id": "anterior",
            "cargo": "Estagiaria de Desenvolvimento",
            "empresa": "Sistemas Ficticios",
            "dataInicioMes": 2,
            "dataInicioAno": 2021,
            "dataFimMes": 12,
            "dataFimAno": 2022,
            "descricao": "- Atuei em modulos de faturamento com Java (Spring Boot) sobre MySQL.",
        },
    ],
    "skills": ["Python", "FastAPI", "PostgreSQL", "Java", "Spring Boot", "MySQL"],
}

VAGA_GERACAO = {
    "titulo": "Desenvolvedora Backend Java",
    "empresa": "Empresa Exemplo",
    "descricao": (
        "Vaga backend com Java, Spring Boot, APIs REST, PostgreSQL e Kubernetes. "
        "Desejavel experiencia com mensageria e AWS."
    ),
}

KEYWORDS_GERACAO = [
    {"termo": "Java", "peso": 1.0},
    {"termo": "Spring Boot", "peso": 0.9},
    {"termo": "APIs REST", "peso": 0.8},
    {"termo": "PostgreSQL", "peso": 0.7},
    {"termo": "Kubernetes", "peso": 0.6},
    {"termo": "AWS", "peso": 0.4},
]

CONTEXTO_GERACAO = [
    {"id": "nota-planos", "tipo": "nota", "factual": False, "titulo": "Planos", "texto": "Quero estudar Kubernetes e AWS no proximo semestre."},
    {"id": "nota-projeto", "tipo": "nota", "factual": True, "titulo": "Projeto", "texto": "Publiquei a API de pedidos em containers Docker no ambiente de homologacao."},
]

KEYWORDS = [
    {"termo": "Python", "peso": 1.0},
    {"termo": "FastAPI", "peso": 0.9},
    {"termo": "PostgreSQL", "peso": 0.8},
    {"termo": "Docker", "peso": 0.7},
    {"termo": "AWS", "peso": 0.5},
]


def _tool(nome: str, descricao: str) -> dict[str, Any]:
    return {
        "name": nome,
        "description": descricao,
        "input_schema": {
            "type": "object",
            "properties": {"oportunidadeId": {"type": "string", "description": "id da oportunidade"}},
            "required": ["oportunidadeId"],
            "additionalProperties": False,
        },
        "strict": True,
    }


TOOLS_SONDA = [
    _tool("buscar_oportunidade", "Le a oportunidade em foco"),
    _tool("analisar_ats", "Analisa a aderencia do perfil a vaga"),
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

    def count_tokens(self, **payload: Any) -> Any:
        return self._alvo.messages.count_tokens(**payload)


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
    turno = TurnRequest.model_validate(
        {
            "oportunidadeId": "op-exemplo",
            "mensagens": [
                {"role": "user", "content": [{"type": "text", "text": "Quais sao os proximos passos para esta vaga?"}]}
            ],
            "tools": TOOLS_SONDA,
        }
    )
    geracao = GenerateCvRequest.model_validate(
        {
            "perfilMestre": PERFIL_GERACAO,
            "vaga": {**VAGA_GERACAO, "keywords": KEYWORDS_GERACAO},
            "keywords": KEYWORDS_GERACAO,
            "contexto": CONTEXTO_GERACAO,
        }
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
        (CHAMADOR_GERACAO, lambda: generate_cv_pipeline(geracao)),
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


class Captura(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.mensagens: list[str] = []

    def emit(self, registro: logging.LogRecord) -> None:
        self.mensagens.append(registro.getMessage())


def _resumo_geracao(rodada: int, op: Any, captura: Captura, resultado: Any, chamadas: list[tuple[Any, float]]) -> None:
    rejeitadas = [m for m in captura.mensagens if m.startswith("frase rejeitada")]
    descartadas = [m for m in captura.mensagens if m.startswith("frase descartada")]
    reparo = next((m for m in captura.mensagens if m.startswith("reparo localizado")), "reparo localizado nao executado")
    cache = [getattr(r.usage, "cache_read_input_tokens", 0) or 0 for r, _ in chamadas]
    print(
        f"{CHAMADOR_GERACAO:<20} rodada={rodada} resumo requisicoes={op.requisicoes} "
        f"rejeitadas={len(rejeitadas)} descartadas={len(descartadas)} cache_lida_por_chamada={cache} | {reparo}"
    )
    for mensagem in rejeitadas + descartadas:
        print(f"{CHAMADOR_GERACAO:<20} rodada={rodada}   {mensagem}")
    if rodada == 1 and resultado is not None and getattr(resultado, "estrutura", None) is not None:
        print(json.dumps(resultado.estrutura.model_dump(by_alias=True), ensure_ascii=False, indent=2))
        print(resultado.markdown)


def _args_validos(args: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(args, dict):
        return False
    propriedades = schema.get("properties", {})
    if any(chave not in propriedades for chave in args):
        return False
    if any(chave not in args for chave in schema.get("required", [])):
        return False
    return all(isinstance(valor, str) for chave, valor in args.items() if propriedades[chave].get("type") == "string")


def _uso(resposta: Any) -> str:
    usage = resposta.usage
    return (
        f"entrada={usage.input_tokens} saida={usage.output_tokens} "
        f"cache_escrita={usage.cache_creation_input_tokens or 0} cache_lida={usage.cache_read_input_tokens or 0}"
    )


def _turno_nativo(gravador: Gravador) -> bool:
    mensagens: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": PEDIDO_NATIVO}]}]
    gravador.chamadas.clear()
    with operacao(operacao_id=f"sonda:{CHAMADOR_NATIVO}"):
        try:
            primeiro = planejar_turno(
                TurnRequest.model_validate({"oportunidadeId": "op-exemplo", "mensagens": mensagens, "tools": TOOLS_SONDA})
            )
        except LLMUnavailable as exc:
            print(f"{CHAMADOR_NATIVO:<20} passo=1 validou=nao ({exc})")
            return False
        usos = [bloco for bloco in primeiro.conteudo if bloco.get("type") == "tool_use"]
        resposta1 = gravador.chamadas[-1][0] if gravador.chamadas else None
        if not usos:
            print(f"{CHAMADOR_NATIVO:<20} passo=1 parada={primeiro.parada} sem tool_use validou=nao")
            return False
        uso = usos[0]
        schema = next(t["input_schema"] for t in TOOLS_SONDA if t["name"] == uso["name"])
        validos = _args_validos(uso.get("input"), schema)
        print(
            f"{CHAMADOR_NATIVO:<20} passo=1 parada={primeiro.parada} tool={uso['name']} "
            f"args={json.dumps(uso.get('input'), ensure_ascii=False)} args_validos={'sim' if validos else 'nao'} "
            f"{_uso(resposta1) if resposta1 else ''}"
        )
        mensagens += [
            {"role": "assistant", "content": primeiro.conteudo},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": uso["id"], "content": json.dumps(OPORTUNIDADE_NATIVA, ensure_ascii=False)}
                ],
            },
        ]
        try:
            segundo = planejar_turno(
                TurnRequest.model_validate({"oportunidadeId": "op-exemplo", "mensagens": mensagens, "tools": TOOLS_SONDA})
            )
        except LLMUnavailable as exc:
            print(f"{CHAMADOR_NATIVO:<20} passo=2 validou=nao ({exc})")
            return False
    texto = " ".join(bloco.get("text", "") for bloco in segundo.conteudo if bloco.get("type") == "text").strip()
    resposta2 = gravador.chamadas[-1][0]
    lida = resposta2.usage.cache_read_input_tokens or 0
    print(
        f"{CHAMADOR_NATIVO:<20} passo=2 parada={segundo.parada} texto={json.dumps(texto[:160], ensure_ascii=False)} "
        f"{_uso(resposta2)} cache_lida_no_segundo_passo={'sim' if lida > 0 else 'nao'}"
    )
    return validos and bool(texto) and segundo.parada == "end_turn"


def _contagem() -> None:
    contador.reiniciar()
    payload = {
        "system": [{"type": "text", "text": SYSTEM_TURNO}],
        "tools": TOOLS_SONDA,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PEDIDO_NATIVO}]}],
    }
    estimativa = contador.estimar(payload)
    tokens = contador.contar(payload)
    via = "estimativa" if contador.api_indisponivel or os.getenv("AI_CONTAGEM_TOKENS", "api") == "estimativa" else "api"
    print(f"contagem_tokens      via={via} tokens={tokens} estimativa_sem_calibrar={estimativa} fator={contador.fator:.2f}")


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
    logger_geracao = logging.getLogger("app.generate")
    logger_geracao.setLevel(logging.INFO)
    for chamador, executar in _casos():
        for rodada in range(1, RODADAS + 1):
            gravador.chamadas.clear()
            captura = Captura()
            logger_geracao.addHandler(captura)
            resultado = None
            try:
                with operacao(operacao_id=f"sonda:{chamador}:{rodada}") as op:
                    resultado = executar()
                degradacao = getattr(resultado, "degradacao", None)
                validou = "nao (" + degradacao + ")" if degradacao else "sim"
            except LLMUnavailable as exc:
                validou = f"nao ({exc})"
            finally:
                logger_geracao.removeHandler(captura)
            if validou != "sim":
                falhas += 1
            _imprimir(chamador, rodada, validou, list(gravador.chamadas))
            if chamador == CHAMADOR_GERACAO:
                _resumo_geracao(rodada, op, captura, resultado, list(gravador.chamadas))
    if not _turno_nativo(gravador):
        falhas += 1
    _contagem()
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
