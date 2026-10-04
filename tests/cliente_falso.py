import copy
import json
import os
from types import SimpleNamespace

from app import llm


def resposta(
    conteudo,
    stop_reason="end_turn",
    entrada=100,
    saida=20,
    cache_lida=0,
    cache_escrita=0,
    modelo="claude-teste",
    categoria=None,
):
    texto = conteudo if isinstance(conteudo, str) else json.dumps(conteudo, ensure_ascii=False)
    return SimpleNamespace(
        model=modelo,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=categoria) if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="text", text=texto)],
        usage=SimpleNamespace(
            input_tokens=entrada,
            output_tokens=saida,
            cache_read_input_tokens=cache_lida,
            cache_creation_input_tokens=cache_escrita,
        ),
    )


class _Mensagens:
    def __init__(self, dono):
        self._dono = dono

    def create(self, **requisicao):
        self._dono.requisicoes.append(copy.deepcopy(requisicao))
        self._dono.timeouts.append(self._dono._timeout_atual)
        if not self._dono.respostas:
            raise AssertionError("cliente falso sem resposta programada")
        proxima = self._dono.respostas.pop(0)
        if isinstance(proxima, BaseException):
            raise proxima
        return proxima


class ClienteFalso:
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.requisicoes = []
        self.timeouts = []
        self._timeout_atual = None
        self.max_retries = []
        self.messages = _Mensagens(self)

    def with_options(self, timeout=None, max_retries=None, **_):
        self._timeout_atual = timeout
        self.max_retries.append(max_retries)
        return self


class ComClienteFalso:
    def __init__(self, *respostas, modelo="claude-teste"):
        self.cliente = ClienteFalso(*respostas)
        self._modelo = modelo
        self._anterior = None
        self._env = {}

    def __enter__(self):
        self._anterior = llm._cliente
        for chave, valor in {"AI_MODEL": self._modelo, "ANTHROPIC_API_KEY": "sk-teste", "AI_MAX_RETRIES": "0"}.items():
            self._env[chave] = os.environ.get(chave)
            os.environ[chave] = valor
        llm.definir_cliente(self.cliente)
        return self.cliente

    def __exit__(self, *_):
        llm.definir_cliente(self._anterior)
        for chave, valor in self._env.items():
            if valor is None:
                os.environ.pop(chave, None)
            else:
                os.environ[chave] = valor
        return False
