from .passos import executar
from .llm import LLMUnavailable


def handler(event: dict, _context) -> dict:
    try:
        return executar(
            event.get("passo", ""), event.get("payload", {}),
            event.get("prazoMs"), event.get("operacao"),
        )
    except LLMUnavailable as exc:
        if "resposta invalida apos reparo" in str(exc):
            return {"erroDefinitivo": str(exc)}
        raise
