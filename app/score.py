from collections import Counter

from .schemas import Keyword, ScoreBreakdown, ScoreResponse
from .text import normalize, tokens

PESO_KEYWORD_MATCH = 0.6
PESO_DENSIDADE = 0.2
PESO_SECOES = 0.2
TETO_OCORRENCIAS = 3

SECOES = {
    "resumo": ("resumo", "summary", "objetivo"),
    "experiencia": ("experiencia", "experience", "profissional"),
    "formacao": ("formacao", "education", "academica"),
    "skills": ("skills", "competencias", "habilidades"),
    "contato": ("contato", "contact", "email"),
}


def _presenca_secoes(markdown_norm: str) -> tuple[int, list[str]]:
    presentes = 0
    faltando = []
    for nome, termos in SECOES.items():
        if any(t in markdown_norm for t in termos):
            presentes += 1
        else:
            faltando.append(nome)
    return presentes, faltando


def calcular_score(markdown: str, keywords: list[Keyword]) -> ScoreResponse:
    markdown_norm = normalize(markdown)
    contagem = Counter(tokens(markdown))

    peso_total = sum(k.peso for k in keywords)
    peso_presente = 0.0
    ocorrencias_capadas = 0
    faltando_keywords = []
    for k in keywords:
        termo_norm = normalize(k.termo)
        ocorr = sum(contagem[t] for t in termo_norm.split())
        if ocorr > 0:
            peso_presente += k.peso
            ocorrencias_capadas += min(ocorr, TETO_OCORRENCIAS)
        else:
            faltando_keywords.append(k.termo)

    keyword_match = peso_presente / peso_total if peso_total > 0 else 0.0
    presentes = len(keywords) - len(faltando_keywords)
    densidade = (
        ocorrencias_capadas / (TETO_OCORRENCIAS * presentes)
        if presentes > 0
        else 0.0
    )
    secoes_presentes, faltando_secoes = _presenca_secoes(markdown_norm)
    secoes = secoes_presentes / len(SECOES)

    total = (
        PESO_KEYWORD_MATCH * keyword_match
        + PESO_DENSIDADE * densidade
        + PESO_SECOES * secoes
    )

    return ScoreResponse(
        score=round(total * 100),
        breakdown=ScoreBreakdown(
            keyword_match=round(keyword_match * 100),
            densidade=round(densidade * 100),
            secoes=round(secoes * 100),
            faltando=faltando_keywords,
        ),
    )
