from collections import Counter

from .llm import LLMUnavailable, complete_model
from .schemas import Keyword, KeywordsResponse
from .text import content_tokens

SYSTEM = (
    "Voce extrai palavras-chave relevantes de descricoes de vaga para "
    "otimizacao ATS. Cada palavra-chave tem um peso de 0 a 1 conforme a "
    "importancia. Responda em JSON."
)


def _user(descricao: str) -> str:
    return (
        "Extraia ate 15 palavras-chave da descricao abaixo e devolva JSON no "
        'formato {"keywords":[{"termo":"...","peso":0.0}]}.\n\n'
        f"Descricao:\n{descricao}"
    )


def _deterministic(descricao: str) -> list[Keyword]:
    counts = Counter(content_tokens(descricao))
    if not counts:
        return []
    mais_comum = counts.most_common(15)
    maior = mais_comum[0][1]
    return [
        Keyword(termo=termo, peso=round(max(0.1, freq / maior), 2))
        for termo, freq in mais_comum
    ]


def extract_keywords(descricao: str) -> list[Keyword]:
    try:
        res = complete_model(SYSTEM, _user(descricao), KeywordsResponse)
        if res.keywords:
            return res.keywords
    except LLMUnavailable:
        pass
    return _deterministic(descricao)
