import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import llm
from app.llm import Operacao

RAIZ = Path(__file__).resolve().parent
PRECOS_PADRAO_MTOK = {"entrada": 2.0, "saida": 10.0, "cache_escrita": 2.5, "cache_lida": 0.2}
VARIAVEIS_PRECO = {
    "entrada": "EVAL_PRECO_ENTRADA_MTOK",
    "saida": "EVAL_PRECO_SAIDA_MTOK",
    "cache_escrita": "EVAL_PRECO_CACHE_ESCRITA_MTOK",
    "cache_lida": "EVAL_PRECO_CACHE_LIDA_MTOK",
}


class ConfiguracaoAusente(RuntimeError):
    pass


def ler_json(caminho: Path) -> Any:
    return json.loads(caminho.read_text(encoding="utf-8"))


def gravar_json(caminho: Path, dados: Any) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def precos_mtok() -> dict[str, float]:
    precos = dict(PRECOS_PADRAO_MTOK)
    for chave, variavel in VARIAVEIS_PRECO.items():
        try:
            precos[chave] = float(os.environ[variavel])
        except (KeyError, ValueError):
            continue
    return precos


@dataclass
class Uso:
    requisicoes: int = 0
    entrada: int = 0
    saida: int = 0
    cache_lida: int = 0
    cache_escrita: int = 0

    def somar_operacao(self, op: Operacao) -> None:
        self.requisicoes += op.requisicoes
        self.entrada += op.uso.entrada
        self.saida += op.uso.saida
        self.cache_lida += op.uso.cache_lida
        self.cache_escrita += op.uso.cache_escrita

    def somar(self, outro: "Uso") -> None:
        self.requisicoes += outro.requisicoes
        self.entrada += outro.entrada
        self.saida += outro.saida
        self.cache_lida += outro.cache_lida
        self.cache_escrita += outro.cache_escrita

    def custo_usd(self) -> float:
        precos = precos_mtok()
        total = (
            self.entrada * precos["entrada"]
            + self.saida * precos["saida"]
            + self.cache_escrita * precos["cache_escrita"]
            + self.cache_lida * precos["cache_lida"]
        )
        return round(total / 1_000_000, 6)

    def como_dict(self) -> dict[str, Any]:
        return {
            "requisicoes": self.requisicoes,
            "entrada": self.entrada,
            "saida": self.saida,
            "cacheLida": self.cache_lida,
            "cacheEscrita": self.cache_escrita,
            "custoEstimadoUsd": self.custo_usd(),
        }


@contextmanager
def medir(uso: Uso) -> Iterator[Operacao]:
    with llm.operacao() as op:
        try:
            yield op
        finally:
            uso.somar_operacao(op)


@dataclass
class Requisicao:
    corpo: dict[str, Any]
    resposta: Any = None


@dataclass
class Gravador:
    alvo: Any
    requisicoes: list[Requisicao] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = _MensagensGravadas(self)

    def with_options(self, **opcoes: Any) -> "Gravador":
        return Gravador(self.alvo.with_options(**opcoes), self.requisicoes)


class _MensagensGravadas:
    def __init__(self, dono: Gravador) -> None:
        self._dono = dono

    def create(self, **corpo: Any) -> Any:
        registro = Requisicao(corpo=corpo)
        self._dono.requisicoes.append(registro)
        registro.resposta = self._dono.alvo.messages.create(**corpo)
        return registro.resposta

    def stream(self, **corpo: Any) -> Any:
        return self._dono.alvo.messages.stream(**corpo)

    def count_tokens(self, **corpo: Any) -> Any:
        return self._dono.alvo.messages.count_tokens(**corpo)


@contextmanager
def gravando() -> Iterator[Gravador]:
    anterior = llm.cliente()
    gravador = Gravador(anterior)
    llm.definir_cliente(gravador)
    try:
        yield gravador
    finally:
        llm.definir_cliente(anterior)


def exigir_credencial() -> None:
    faltando = [nome for nome in ("ANTHROPIC_API_KEY", "AI_MODEL") if not os.getenv(nome, "").strip()]
    if faltando:
        raise ConfiguracaoAusente(
            "defina " + " e ".join(faltando) + " para rodar ao vivo, ou use --falso para provar o executor"
        )


def media(valores: list[float]) -> float | None:
    validos = [v for v in valores if v is not None]
    return round(sum(validos) / len(validos), 4) if validos else None


@dataclass(frozen=True)
class Regressao:
    metrica: str
    atual: float | None
    referencia: float | None
    motivo: str

    def como_dict(self) -> dict[str, Any]:
        return {"metrica": self.metrica, "atual": self.atual, "referencia": self.referencia, "motivo": self.motivo}


def comparar(resumo: dict[str, Any], regras: dict[str, dict[str, Any]]) -> list[Regressao]:
    regressoes: list[Regressao] = []
    for metrica, regra in regras.items():
        atual = resumo.get(metrica)
        maior = regra["direcao"] == "maior"
        absoluto = regra.get("absoluto")
        if atual is None:
            if absoluto is not None or regra.get("valor") is not None:
                regressoes.append(Regressao(metrica, None, regra.get("valor"), "metrica ausente no resultado"))
            continue
        if absoluto is not None and (atual < absoluto if maior else atual > absoluto):
            limite = "abaixo do minimo" if maior else "acima do maximo"
            regressoes.append(Regressao(metrica, atual, absoluto, f"{limite} absoluto {absoluto}"))
            continue
        valor = regra.get("valor")
        if valor is None:
            continue
        queda = valor - atual if maior else atual - valor
        if queda > regra.get("limiar", 0) + 1e-9:
            regressoes.append(
                Regressao(metrica, atual, valor, f"caiu {round(queda, 4)} alem do limiar {regra.get('limiar', 0)}")
            )
    return regressoes


def regras_atualizadas(resumo: dict[str, Any], regras: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {metrica: {**regra, "valor": resumo.get(metrica)} for metrica, regra in regras.items()}
