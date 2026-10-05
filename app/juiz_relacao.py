import os
from dataclasses import dataclass

from pydantic import Field

from .carregador_prompts import obter as obter_prompt
from .llm import ValidacaoSemantica, complete_model
from .schemas import CamelModel

PROMPT = "juiz_relacao"
CHAMADOR = "juiz_relacao"
ESFORCO = "low"
VARIAVEL = "AI_JUIZ_RELACAO"
VALORES_DESLIGADO = ("0", "false", "nao", "off")


class NotaRelacao(CamelModel):
    chave: str = Field(description="A chave da frase, copiada sem alteracao")
    relacao_sustentada: bool
    justificativa: str


class JulgamentoRelacao(CamelModel):
    notas: list[NotaRelacao]


@dataclass(frozen=True)
class FraseParaJulgar:
    chave: str
    texto: str
    fontes: tuple[tuple[str, str], ...]


def ligado() -> bool:
    return os.environ.get(VARIAVEL, "1").strip().lower() not in VALORES_DESLIGADO


def versao() -> str:
    return obter_prompt(PROMPT).rotulo


def _bloco(frase: FraseParaJulgar) -> str:
    fontes = "\n".join(
        f"    [{fonte_id}] {linha}" if indice == 0 else f"      {linha}"
        for fonte_id, texto in frase.fontes
        for indice, linha in enumerate(texto.splitlines() or [""])
    )
    return f"- chave: {frase.chave}\n  frase: {frase.texto}\n  fontes citadas:\n{fontes or '    nenhuma'}"


def montar_user(frases: list[FraseParaJulgar]) -> str:
    return "Frases para conferir:\n\n" + "\n\n".join(_bloco(frase) for frase in frases)


def julgar(frases: list[FraseParaJulgar]) -> dict[str, NotaRelacao]:
    if not frases:
        return {}
    esperadas = {frase.chave for frase in frases}

    def validar(resposta: JulgamentoRelacao) -> None:
        faltando = esperadas - {nota.chave.strip() for nota in resposta.notas}
        if faltando:
            raise ValidacaoSemantica("faltam notas para as chaves: " + ", ".join(sorted(faltando)))

    resposta = complete_model(
        obter_prompt(PROMPT).texto,
        montar_user(frases),
        JulgamentoRelacao,
        chamador=CHAMADOR,
        esforco=ESFORCO,
        validar=validar,
        prompt_version=versao(),
    )
    return {nota.chave.strip(): nota for nota in resposta.notas if nota.chave.strip() in esperadas}
