import json
import os
import re
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_STRICT_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    }
)


class LLMUnavailable(Exception):
    pass


def _provider() -> str:
    return os.getenv("AI_PROVIDER", "groq").lower()


def _env_float(nome: str, fallback: float) -> float:
    try:
        valor = float(os.getenv(nome, str(fallback)))
    except ValueError:
        return fallback
    return valor if valor > 0 else fallback


def _client_and_model() -> tuple[OpenAI, str]:
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY", ""),
        timeout=_env_float("AI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
    ), os.getenv("AI_MODEL", "openai/gpt-oss-120b")


def _supports_reasoning_effort(model: str) -> bool:
    normalized = model.lower()
    return any(
        marker in normalized
        for marker in ("gpt-oss", "qwen", "deepseek", "reasoning")
    )


def _strict_models() -> set[str]:
    configurados = os.getenv("AI_STRICT_MODELS", "")
    if configurados.strip():
        return {item.strip().lower() for item in configurados.split(",") if item.strip()}
    return set(DEFAULT_STRICT_MODELS)


def _strict_schema(valor: object, obrigatorio: bool = True) -> object:
    if isinstance(valor, list):
        return [_strict_schema(item, obrigatorio) for item in valor]
    if not isinstance(valor, dict):
        return valor

    resultado = {
        chave: _strict_schema(item, obrigatorio)
        for chave, item in valor.items()
        if chave not in {"default", "title", "$schema"}
    }
    propriedades = resultado.get("properties")
    if isinstance(propriedades, dict):
        required_atual = set(resultado.get("required", []))
        for nome, propriedade in list(propriedades.items()):
            if nome not in required_atual and isinstance(propriedade, dict):
                if "anyOf" not in propriedade and "oneOf" not in propriedade:
                    propriedades[nome] = {"anyOf": [propriedade, {"type": "null"}]}
        resultado["required"] = list(propriedades)
        resultado["additionalProperties"] = False
    return resultado


def _response_format(model: str, schema: type[T]) -> dict[str, object]:
    modo = os.getenv("AI_JSON_SCHEMA_MODE", "auto").lower()
    usar_strict = modo == "strict" or (modo == "auto" and model.lower() in _strict_models())
    if not usar_strict or modo == "object":
        return {"type": "json_object"}
    nome = re.sub(r"[^a-zA-Z0-9_-]+", "_", schema.__name__) or "response"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": nome,
            "strict": True,
            "schema": _strict_schema(schema.model_json_schema()),
        },
    }


def complete_model(
    system: str, user: str, schema: type[T], retries: int = 2
) -> T:
    if _provider() != "groq":
        raise LLMUnavailable("somente Groq esta habilitado para geracao")
    if not os.getenv("GROQ_API_KEY"):
        raise LLMUnavailable("GROQ_API_KEY ausente")

    client, model = _client_and_model()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last: Exception | None = None
    max_completion_tokens = int(os.getenv("AI_MAX_COMPLETION_TOKENS", "8192"))
    reasoning_effort = os.getenv("AI_REASONING_EFFORT", "medium")
    strict_fallback = False
    for tentativa in range(retries + 1):
        try:
            params = {
                "model": model,
                "messages": messages,
                "response_format": (
                    {"type": "json_object"}
                    if strict_fallback
                    else _response_format(model, schema)
                ),
                "temperature": 0,
                "max_completion_tokens": max_completion_tokens,
            }
            if _supports_reasoning_effort(model):
                params["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**params)
            content = resp.choices[0].message.content or "{}"
            return schema.model_validate(json.loads(content))
        except Exception as exc:
            last = exc
            if (
                tentativa == 0
                and not strict_fallback
                and _response_format(model, schema).get("type") == "json_schema"
            ):
                strict_fallback = True
                continue
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "A resposta anterior era inválida. Responda APENAS com "
                        "JSON válido exatamente no formato pedido, sem texto extra."
                    ),
                }
            )
    raise LLMUnavailable(str(last))
