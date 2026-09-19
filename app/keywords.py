from .llm import LLMUnavailable, complete_model
from .schemas import Keyword, KeywordsLlmResponse

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


def extract_keywords(descricao: str) -> list[Keyword]:
    res = complete_model(SYSTEM, _user(descricao), KeywordsLlmResponse)
    if not res.keywords:
        raise LLMUnavailable("extracao de keywords retornou lista vazia")
    return res.keywords
