import json
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    pass


def _provider() -> str:
    return os.getenv("AI_PROVIDER", "groq").lower()


def _client_and_model() -> tuple[OpenAI, str]:
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY", ""),
    ), os.getenv("AI_MODEL", "openai/gpt-oss-20b")


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
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = resp.choices[0].message.content or "{}"
            return schema.model_validate(json.loads(content))
        except Exception as exc:
            last = exc
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
