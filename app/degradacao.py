import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Degradacao:
    codigo: str
    frase: str


KEYWORDS_INDISPONIVEIS = Degradacao(
    "keywords_llm_indisponivel",
    "A extração de keywords da vaga está indisponível no momento. Tente novamente em instantes.",
)
REESCRITA_INDISPONIVEL = Degradacao(
    "reescrita_llm_indisponivel",
    "A reescrita está indisponível no momento; entregamos uma versão montada diretamente do seu perfil-mestre.",
)
REESCRITA_REJEITADA = Degradacao(
    "reescrita_rejeitada_validacao",
    "A reescrita não passou na verificação de fatos e formato; entregamos uma versão montada diretamente do seu perfil-mestre.",
)
AJUSTE_INDISPONIVEL = Degradacao(
    "ajuste_aderencia_llm_indisponivel",
    "O ajuste extra de aderência ficou indisponível; mantivemos a versão já validada.",
)
AJUSTE_REJEITADO = Degradacao(
    "ajuste_aderencia_rejeitado_validacao",
    "O ajuste extra de aderência não passou na verificação de fatos; mantivemos a versão já validada.",
)
CORTE_INDISPONIVEL = Degradacao(
    "corte_pagina_llm_indisponivel",
    "A redução para uma página ficou indisponível; mantivemos a versão anterior.",
)
CORTE_REJEITADO = Degradacao(
    "corte_pagina_rejeitado_validacao",
    "Não foi possível reduzir para uma página sem perder fatos; mantivemos a versão anterior.",
)
COPILOTO_INDISPONIVEL = Degradacao(
    "copiloto_turno_indisponivel",
    "O copiloto está indisponível no momento. Tente novamente em instantes.",
)
REDACAO_INDISPONIVEL = Degradacao(
    "redacao_indisponivel",
    "A redação está indisponível no momento. Tente novamente em instantes.",
)


def registrar(degradacao: Degradacao, causa: object) -> str:
    logger.warning("degradacao codigo=%s causa=%s", degradacao.codigo, causa)
    return degradacao.frase
