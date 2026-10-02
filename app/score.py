from .casamento import canonico, termo_presente
from .schemas import Keyword, ScoreBreakdown, ScoreResponse
from .secoes import SECOES_OBRIGATORIAS, secoes_reconhecidas

SCORE_VERSAO = 2
PESO_COBERTURA = 0.6
PESO_EXPERIENCIA = 0.2
PESO_SECOES = 0.2


def _keywords_unicas(keywords: list[Keyword]) -> list[Keyword]:
    por_canonico: dict[str, Keyword] = {}
    for k in keywords:
        if not k.termo.strip():
            continue
        chave = canonico(k.termo).lower()
        atual = por_canonico.get(chave)
        if atual is None or k.peso > atual.peso:
            por_canonico[chave] = k
    return list(por_canonico.values())


def calcular_score(markdown: str, keywords: list[Keyword]) -> ScoreResponse:
    secoes = secoes_reconhecidas(markdown)
    experiencia = secoes.get("experiencia", "")
    unicas = _keywords_unicas(keywords)

    peso_total = sum(k.peso for k in unicas)
    peso_presente = 0.0
    peso_em_experiencia = 0.0
    faltando_keywords = []
    for k in unicas:
        if termo_presente(k.termo, markdown):
            peso_presente += k.peso
            if termo_presente(k.termo, experiencia):
                peso_em_experiencia += k.peso
        else:
            faltando_keywords.append(k.termo)

    cobertura = peso_presente / peso_total if peso_total > 0 else 0.0
    sustentacao = peso_em_experiencia / peso_total if peso_total > 0 else 0.0
    presentes = sum(1 for secao in SECOES_OBRIGATORIAS if secao in secoes)
    estrutura = presentes / len(SECOES_OBRIGATORIAS)

    total = (
        PESO_COBERTURA * cobertura
        + PESO_EXPERIENCIA * sustentacao
        + PESO_SECOES * estrutura
    )

    return ScoreResponse(
        score=round(total * 100),
        score_versao=SCORE_VERSAO,
        breakdown=ScoreBreakdown(
            keyword_match=round(cobertura * 100),
            densidade=round(sustentacao * 100),
            secoes=round(estrutura * 100),
            faltando=faltando_keywords,
        ),
    )
