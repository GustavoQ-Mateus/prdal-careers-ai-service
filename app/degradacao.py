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
