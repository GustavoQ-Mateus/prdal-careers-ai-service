import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from . import telemetria
from .seguranca import em_desenvolvimento

T = TypeVar("T", bound=BaseModel)

ESFORCOS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS_PADRAO = 16000
TIMEOUT_TETO_PADRAO_S = 120.0
TIMEOUT_PISO_PADRAO_S = 5.0
MAX_RETRIES_PADRAO = 2
ESPERA_INICIAL_S = 0.5
ESPERA_MAXIMA_S = 8.0
STATUS_REPETIVEIS = (408, 409, 429)

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


class PrazoEsgotado(Exception):
    pass


class TetoDeRequisicoes(LLMUnavailable):
    pass


class ValidacaoSemantica(ValueError):
    pass


class ModeloAusente(RuntimeError):
    pass


def _env_float(nome: str, fallback: float) -> float:
    try:
        valor = float(os.getenv(nome, str(fallback)))
    except ValueError:
        return fallback
    return valor if valor > 0 else fallback


def _env_int(nome: str, fallback: int) -> int:
    try:
        valor = int(os.getenv(nome, str(fallback)))
    except ValueError:
        return fallback
    return valor if valor >= 0 else fallback


def modelo_configurado() -> str | None:
    return os.getenv("AI_MODEL", "").strip() or None


def exigir_modelo_no_boot() -> None:
    if em_desenvolvimento() or modelo_configurado():
        return
    raise ModeloAusente(
        "AI_MODEL ausente; defina o modelo do Claude antes de subir o ai-service "
        "fora de PRDAL_AMBIENTE=desenvolvimento"
    )


def _teto_s() -> float:
    return _env_float("AI_TIMEOUT_TETO_S", TIMEOUT_TETO_PADRAO_S)


def _piso_s() -> float:
    return min(_env_float("AI_TIMEOUT_PISO_S", TIMEOUT_PISO_PADRAO_S), _teto_s())


@dataclass
class Uso:
    entrada: int = 0
    saida: int = 0
    cache_lida: int = 0
    cache_escrita: int = 0
    chamadas: int = 0

    def somar(self, usage: Any) -> None:
        self.entrada += getattr(usage, "input_tokens", 0) or 0
        self.saida += getattr(usage, "output_tokens", 0) or 0
        self.cache_lida += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_escrita += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.chamadas += 1


@dataclass
class Operacao:
    prazo: float | None
    operacao_id: str | None = None
    uso: Uso = field(default_factory=Uso)
    modelo: str | None = None
    requisicoes: int = 0
    teto_requisicoes: int | None = None

    def restante_s(self) -> float | None:
        if self.prazo is None:
            return None
        return self.prazo - time.monotonic()


_operacao: ContextVar[Operacao | None] = ContextVar("operacao_llm", default=None)


@contextmanager
def operacao(prazo_ms: int | None = None, operacao_id: str | None = None) -> Iterator[Operacao]:
    if prazo_ms is not None and prazo_ms <= 0:
        raise PrazoEsgotado("prazo da operacao ja esgotado")
    atual = Operacao(
        prazo=time.monotonic() + prazo_ms / 1000 if prazo_ms is not None else None,
        operacao_id=operacao_id,
    )
    token = _operacao.set(atual)
    try:
        yield atual
    finally:
        _operacao.reset(token)


def operacao_atual() -> Operacao:
    return _operacao.get() or Operacao(prazo=None)


@contextmanager
def teto_de_requisicoes(teto: int) -> Iterator[Operacao]:
    atual = _operacao.get()
    if atual is None:
        with operacao() as nova:
            nova.teto_requisicoes = teto
            yield nova
        return
    anterior = atual.teto_requisicoes
    atual.teto_requisicoes = atual.requisicoes + teto
    try:
        yield atual
    finally:
        atual.teto_requisicoes = anterior


def _contar_requisicao(op: Operacao) -> None:
    if op.teto_requisicoes is not None and op.requisicoes >= op.teto_requisicoes:
        raise TetoDeRequisicoes(f"teto de {op.teto_requisicoes} requisicoes ao modelo atingido")
    op.requisicoes += 1


def timeout_da_chamada(op: Operacao) -> float:
    restante = op.restante_s()
    if restante is None:
        return _teto_s()
    if restante <= 0 or restante < _piso_s():
        raise PrazoEsgotado("prazo da operacao esgotado antes da chamada ao modelo")
    return min(restante, _teto_s())


def _tentativas_maximas() -> int:
    return 1 + _env_int("AI_MAX_RETRIES", MAX_RETRIES_PADRAO)


def _deve_repetir(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    if not isinstance(exc, anthropic.APIStatusError):
        return False
    pedido = exc.response.headers.get("x-should-retry")
    if pedido in ("true", "false"):
        return pedido == "true"
    return exc.status_code in STATUS_REPETIVEIS or exc.status_code >= 500


def _espera_pedida_s(exc: Exception) -> float | None:
    if not isinstance(exc, anthropic.APIStatusError):
        return None
    cabecalhos = exc.response.headers
    for nome, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            valor = float(cabecalhos.get(nome, ""))
        except ValueError:
            continue
        if valor > 0:
            return valor / divisor
    return None


def _espera_antes_de_repetir(exc: Exception, repeticao: int) -> float:
    pedida = _espera_pedida_s(exc)
    if pedida is not None:
        return pedida
    return min(ESPERA_INICIAL_S * 2**repeticao, ESPERA_MAXIMA_S)


def _cabe_nova_tentativa(op: Operacao, espera: float) -> bool:
    restante = op.restante_s()
    if restante is None:
        return espera <= _teto_s()
    return restante - espera >= _piso_s()


_cliente: Any = None
_trava_cliente = threading.Lock()


def _credencial_presente() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip() or os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip())


def cliente() -> Any:
    global _cliente
    if _cliente is None:
        with _trava_cliente:
            if _cliente is None:
                _cliente = anthropic.Anthropic(max_retries=0, timeout=_teto_s())
    return _cliente


def definir_cliente(novo: Any) -> None:
    global _cliente
    with _trava_cliente:
        _cliente = novo


def montar_requisicao(
    modelo: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    esforco: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": modelo,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        "output_config": {
            "effort": esforco,
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
        },
    }


def _motivo_falha(exc: Exception) -> str:
    if isinstance(exc, anthropic.APITimeoutError):
        return "tempo esgotado na chamada ao modelo"
    if isinstance(exc, anthropic.RateLimitError):
        return "limite de requisicoes do provedor atingido"
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "credencial recusada pelo provedor"
    if isinstance(exc, anthropic.BadRequestError):
        return f"requisicao recusada pelo provedor: {exc.message}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"erro do provedor status={exc.status_code}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "falha de conexao com o provedor"
    return f"falha no cliente do provedor: {type(exc).__name__}"


RESULTADOS_POR_PARADA = {"refusal": "recusa", "max_tokens": "limite"}


def _atributos(requisicao: dict[str, Any], op: Operacao, rotulo: dict[str, Any]) -> dict[str, Any]:
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.model": requisicao["model"],
        "gen_ai.request.max_tokens": requisicao["max_tokens"],
        "prdal.esforco": requisicao["output_config"]["effort"],
        "prdal.operacao_id": op.operacao_id,
        **rotulo,
    }


def _chamar(requisicao: dict[str, Any], op: Operacao, rotulo: dict[str, Any]) -> Any:
    maximo = _tentativas_maximas()
    repeticao = 0
    while True:
        timeout = timeout_da_chamada(op)
        _contar_requisicao(op)
        try:
            return _tentar(requisicao, op, rotulo, timeout)
        except anthropic.AnthropicError as exc:
            if repeticao + 1 >= maximo or not _deve_repetir(exc):
                raise LLMUnavailable(_motivo_falha(exc)) from exc
            espera = _espera_antes_de_repetir(exc, repeticao)
            if not _cabe_nova_tentativa(op, espera):
                raise LLMUnavailable(_motivo_falha(exc)) from exc
            time.sleep(espera)
            repeticao += 1


def _tentar(requisicao: dict[str, Any], op: Operacao, rotulo: dict[str, Any], timeout: float) -> Any:
    atributos = _atributos(requisicao, op, rotulo)
    inicio_ns = time.time_ns()
    inicio = time.perf_counter()
    try:
        resposta = cliente().with_options(timeout=timeout, max_retries=0).messages.create(**requisicao)
    except anthropic.AnthropicError as exc:
        motivo = _motivo_falha(exc)
        logger.warning("chamada ao modelo falhou tipo=%s motivo=%s", type(exc).__name__, motivo)
        atributos.update(
            {
                "prdal.resultado": "erro",
                "error.type": type(exc).__name__,
                "prdal.latencia_ms": round((time.perf_counter() - inicio) * 1000),
            }
        )
        telemetria.registrar_chamada(atributos, inicio_ns, time.time_ns(), motivo)
        raise
    usage = getattr(resposta, "usage", None)
    op.uso.somar(usage)
    op.modelo = getattr(resposta, "model", None) or requisicao["model"]
    entrada = getattr(usage, "input_tokens", 0) or 0
    cache_lida = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_escrita = getattr(usage, "cache_creation_input_tokens", 0) or 0
    atributos.update(
        {
            "gen_ai.response.model": op.modelo,
            "gen_ai.response.finish_reasons": (str(resposta.stop_reason),),
            "gen_ai.usage.input_tokens": entrada + cache_lida + cache_escrita,
            "gen_ai.usage.output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "gen_ai.usage.cache_read.input_tokens": cache_lida,
            "gen_ai.usage.cache_creation.input_tokens": cache_escrita,
            "prdal.resultado": RESULTADOS_POR_PARADA.get(resposta.stop_reason, "ok"),
            "prdal.latencia_ms": round((time.perf_counter() - inicio) * 1000),
        }
    )
    telemetria.registrar_chamada(atributos, inicio_ns, time.time_ns())
    return resposta


def _texto_da_resposta(resposta: Any) -> str:
    if resposta.stop_reason == "refusal":
        detalhe = getattr(resposta, "stop_details", None)
        categoria = getattr(detalhe, "category", None) if detalhe else None
        raise LLMUnavailable(f"o modelo recusou a resposta categoria={categoria or 'nao informada'}")
    if resposta.stop_reason == "max_tokens":
        raise LLMUnavailable("resposta do modelo truncada no limite de tokens")
    return "".join(bloco.text for bloco in resposta.content if bloco.type == "text")


def _validar(texto: str, schema: type[T], validar: Callable[[T], None] | None) -> T:
    resultado = schema.model_validate_json(texto)
    if validar:
        validar(resultado)
    return resultado


def _erro_legivel(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(parte) for parte in erro['loc']) or 'resposta'}: {erro['msg']}"
            for erro in exc.errors()
        )
    return str(exc)


def versao_do_prompt(chamador: str, system: str) -> str:
    return f"{chamador}.{hashlib.sha256(system.encode('utf-8')).hexdigest()[:12]}"


def complete_model(
    system: str,
    user: str,
    schema: type[T],
    *,
    chamador: str,
    esforco: str,
    validar: Callable[[T], None] | None = None,
    max_tokens: int | None = None,
    prompt_version: str | None = None,
) -> T:
    if esforco not in ESFORCOS:
        raise ValueError(f"esforco invalido: {esforco}")
    modelo = modelo_configurado()
    if not modelo:
        raise LLMUnavailable("AI_MODEL ausente")
    if not _credencial_presente():
        raise LLMUnavailable("ANTHROPIC_API_KEY ausente")
    op = operacao_atual()
    requisicao = montar_requisicao(
        modelo,
        system,
        user,
        schema,
        esforco,
        max_tokens or _env_int("AI_MAX_TOKENS", MAX_TOKENS_PADRAO),
    )
    rotulo = {
        "prdal.chamador": chamador,
        "prdal.prompt_version": prompt_version or versao_do_prompt(chamador, system),
        "prdal.tentativa": 1,
    }
    resposta = _chamar(requisicao, op, rotulo)
    texto = _texto_da_resposta(resposta)
    try:
        return _validar(texto, schema, validar)
    except (ValidationError, ValidacaoSemantica) as exc:
        erro = _erro_legivel(exc)
        logger.warning("validacao da resposta falhou chamador=%s erro=%s", chamador, erro)
    requisicao["messages"] = [
        *requisicao["messages"],
        {"role": "assistant", "content": [{"type": "text", "text": texto}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "A resposta anterior nao passou na validacao: "
                        f"{erro}. Corrija apenas o que o erro aponta e devolva a resposta completa."
                    ),
                }
            ],
        },
    ]
    resposta = _chamar(requisicao, op, {**rotulo, "prdal.tentativa": 2})
    texto = _texto_da_resposta(resposta)
    try:
        return _validar(texto, schema, validar)
    except (ValidationError, ValidacaoSemantica) as exc:
        raise LLMUnavailable(f"resposta invalida apos reparo: {_erro_legivel(exc)}") from exc
