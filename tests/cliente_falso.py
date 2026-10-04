import copy
import json
import os
import time
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


def resposta_blocos(blocos, stop_reason="end_turn", entrada=100, saida=20, cache_lida=0, cache_escrita=0):
    base = resposta("", stop_reason, entrada, saida, cache_lida, cache_escrita)
    base.content = [SimpleNamespace(**bloco) for bloco in blocos]
    return base


class StreamFalso:
    def __init__(self, textos, final, falha_apos=None, falha=None, atraso=0.0, entrada=100):
        self.textos = list(textos)
        self.final = final
        self.falha_apos = falha_apos
        self.falha = falha
        self.atraso = atraso
        self.entrada = entrada
        self.instantes = []
        self.fechado = False
        self._snapshot = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechado = True
        return False

    @property
    def current_message_snapshot(self):
        if self._snapshot is None:
            raise AssertionError("sem snapshot")
        return self._snapshot

    def __iter__(self):
        self._snapshot = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=self.entrada, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        )
        yield SimpleNamespace(type="message_start")
        blocos = self.textos if self.textos and isinstance(self.textos[0], list) else [self.textos]
        emitidos = 0
        for bloco in blocos:
            yield SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text"))
            for texto in bloco:
                if self.falha_apos is not None and emitidos >= self.falha_apos:
                    raise self.falha
                time.sleep(self.atraso)
                self.instantes.append(time.monotonic())
                emitidos += 1
                yield SimpleNamespace(type="text", text=texto, snapshot="")
        if self.falha_apos is not None and emitidos >= self.falha_apos:
            raise self.falha

    def get_final_message(self):
        return self.final


class _Mensagens:
    def __init__(self, dono):
        self._dono = dono

    def stream(self, **requisicao):
        self._dono.requisicoes.append(copy.deepcopy(requisicao))
        self._dono.timeouts.append(self._dono._timeout_atual)
        if not self._dono.respostas:
            raise AssertionError("cliente falso sem stream programado")
        proximo = self._dono.respostas.pop(0)
        if isinstance(proximo, BaseException):
            raise proximo
        return proximo

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
