import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass

from . import rag
from .carregador_prompts import PROMPTS_OBRIGATORIOS, PromptAusente, obter
from .llm import _credencial_presente, modelo_configurado
from .observabilidade import request_id_atual

logger = logging.getLogger("prdal.prontidao")


@dataclass(frozen=True)
class SituacaoDependencia:
    nome: str
    obrigatoria: bool
    estado: str
    detalhe: str | None = None

    def como_dict(self) -> dict[str, object]:
        return {chave: valor for chave, valor in asdict(self).items() if chave != "detalhe"}


def _verificar(nome: str, obrigatoria: bool, sonda: Callable[[], None]) -> SituacaoDependencia:
    try:
        sonda()
    except Exception as exc:
        return SituacaoDependencia(nome, obrigatoria, "indisponivel", str(exc) or type(exc).__name__)
    return SituacaoDependencia(nome, obrigatoria, "ok")


def _prompts() -> None:
    for prompt_id in PROMPTS_OBRIGATORIOS:
        try:
            obter(prompt_id)
        except (KeyError, PromptAusente) as exc:
            raise RuntimeError(f"prompt {prompt_id} ausente") from exc


def _claude() -> None:
    if not modelo_configurado():
        raise RuntimeError("AI_MODEL ausente")
    if not _credencial_presente():
        raise RuntimeError("ANTHROPIC_API_KEY ausente")


def _embeddings() -> None:
    if not rag._model.cache_info().currsize:
        raise RuntimeError("modelo de embeddings nao carregado")


def prontidao() -> tuple[bool, dict[str, object]]:
    dependencias = [
        _verificar("prompts", True, _prompts),
        _verificar("claude", True, _claude),
        _verificar("embeddings", False, _embeddings),
    ]
    pronto = all(d.estado == "ok" for d in dependencias if d.obrigatoria)
    for d in dependencias:
        if d.estado != "ok":
            logger.warning(
                "dependencia fora",
                extra={
                    "requestId": request_id_atual(),
                    "dependencia": d.nome,
                    "obrigatoria": d.obrigatoria,
                    "estado": d.estado,
                    "detalhe": d.detalhe,
                },
            )
    return pronto, {
        "servico": "ai-service",
        "status": "pronto" if pronto else "indisponivel",
        "dependencias": [d.como_dict() for d in dependencias],
    }
