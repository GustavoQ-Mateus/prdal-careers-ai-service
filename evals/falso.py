import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from app import llm
from app.contexto import contador

USO_FALSO = {"entrada": 900, "saida": 120, "cache_lida": 0, "cache_escrita": 0}


def resposta_falsa(roteiro: dict[str, Any]) -> SimpleNamespace:
    if "json" in roteiro:
        blocos = [{"type": "text", "text": json.dumps(roteiro["json"], ensure_ascii=False)}]
    else:
        blocos = roteiro["blocos"]
    uso = {**USO_FALSO, **roteiro.get("uso", {})}
    parada = roteiro.get("parada") or ("tool_use" if any(b["type"] == "tool_use" for b in blocos) else "end_turn")
    return SimpleNamespace(
        model="claude-falso",
        stop_reason=parada,
        stop_details=None,
        content=[SimpleNamespace(**bloco) for bloco in blocos],
        usage=SimpleNamespace(
            input_tokens=uso["entrada"],
            output_tokens=uso["saida"],
            cache_read_input_tokens=uso["cache_lida"],
            cache_creation_input_tokens=uso["cache_escrita"],
        ),
    )


class ClienteFalso:
    def __init__(self, roteiros: list[dict[str, Any]]) -> None:
        self.pendentes = list(roteiros)
        self.messages = self

    def with_options(self, **_opcoes: Any) -> "ClienteFalso":
        return self

    def create(self, **_corpo: Any) -> SimpleNamespace:
        if not self.pendentes:
            raise AssertionError("roteiro falso sem resposta para esta requisicao")
        return resposta_falsa(self.pendentes.pop(0))

    def count_tokens(self, **_corpo: Any) -> Any:
        raise RuntimeError("contagem indisponivel no cliente falso")


@contextmanager
def cliente_falso(roteiros: list[dict[str, Any]]) -> Iterator[ClienteFalso]:
    anterior = llm._cliente
    ambiente = {"AI_MODEL": "claude-falso", "ANTHROPIC_API_KEY": "falso", "AI_MAX_RETRIES": "0"}
    guardado = {chave: os.environ.get(chave) for chave in ambiente}
    os.environ.update(ambiente)
    cliente = ClienteFalso(roteiros)
    llm.definir_cliente(cliente)
    contador.reiniciar()
    try:
        yield cliente
    finally:
        llm.definir_cliente(anterior)
        contador.reiniciar()
        for chave, valor in guardado.items():
            if valor is None:
                os.environ.pop(chave, None)
            else:
                os.environ[chave] = valor
