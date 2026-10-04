import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from .seguranca import em_desenvolvimento

T = TypeVar("T", bound=BaseModel)

ESFORCOS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS_PADRAO = 16000
TIMEOUT_TETO_PADRAO_S = 120.0
TIMEOUT_PISO_PADRAO_S = 5.0
MAX_RETRIES_PADRAO = 2

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    pass


class PrazoEsgotado(Exception):
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
class Operacao:
    prazo: float | None
    operacao_id: str | None = None

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


def timeout_da_chamada(op: Operacao) -> float:
    restante = op.restante_s()
    if restante is None:
        return _teto_s()
    if restante <= 0:
        raise PrazoEsgotado("prazo da operacao esgotado antes da chamada ao modelo")
    return max(_piso_s(), min(restante, _teto_s()))


_cliente: Any = None
_trava_cliente = threading.Lock()


def _credencial_presente() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip() or os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip())


def cliente() -> Any:
    global _cliente
    if _cliente is None:
        with _trava_cliente:
            if _cliente is None:
                _cliente = anthropic.Anthropic(
                    max_retries=_env_int("AI_MAX_RETRIES", MAX_RETRIES_PADRAO),
                    timeout=_teto_s(),
                )
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
        "system": [{"type": "text", "text": system}],
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


def _chamar(requisicao: dict[str, Any], op: Operacao) -> Any:
    timeout = timeout_da_chamada(op)
    try:
        return cliente().with_options(timeout=timeout).messages.create(**requisicao)
    except anthropic.AnthropicError as exc:
        motivo = _motivo_falha(exc)
        logger.warning("chamada ao modelo falhou tipo=%s motivo=%s", type(exc).__name__, motivo)
        raise LLMUnavailable(motivo) from exc


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


def complete_model(
    system: str,
    user: str,
    schema: type[T],
    *,
    chamador: str,
    esforco: str,
    validar: Callable[[T], None] | None = None,
    max_tokens: int | None = None,
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
    resposta = _chamar(requisicao, op)
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
    resposta = _chamar(requisicao, op)
    texto = _texto_da_resposta(resposta)
    try:
        return _validar(texto, schema, validar)
    except (ValidationError, ValidacaoSemantica) as exc:
        raise LLMUnavailable(f"resposta invalida apos reparo: {_erro_legivel(exc)}") from exc
