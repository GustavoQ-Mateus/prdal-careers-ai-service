from dataclasses import dataclass

from pydantic import Field

from app.llm import ValidacaoSemantica, complete_model
from app.schemas import CamelModel, Keyword

CHAMADOR = "juiz_geracao"
ESFORCO = "medium"

SYSTEM = (
    "Voce avalia frases de um curriculo gerado para uma vaga. Cada frase cita as fontes do perfil "
    "do candidato que a sustentam. Para cada frase, responda duas perguntas de forma independente.\n\n"
    "1. respondeRequisito: a frase responde a um requisito citado da vaga? Verdadeiro so quando a "
    "frase trata de uma habilidade, ferramenta, responsabilidade, dominio ou resultado que a "
    "descricao ou as keywords da vaga pedem. Copie em requisitoCitado o trecho curto da vaga que a "
    "frase atende; deixe vazio quando a resposta for falsa.\n\n"
    "2. relacaoSustentada: toda relacao que a frase afirma entre fatos esta escrita nas fontes "
    "citadas? Considere causa, resultado, autoria e escala. A resposta e falsa quando a frase:\n"
    "- junta dois fatos que a fonte traz separados e afirma que um causou ou produziu o outro, por "
    "exemplo 'migrou o banco para PostgreSQL, reduzindo o tempo em 30%' quando a fonte diz que "
    "migrou o banco e, em outro ponto, que reduziu o tempo em 30%;\n"
    "- atribui ao candidato, sozinho, algo que a fonte atribui ao time ou a outra pessoa;\n"
    "- aumenta numero, escala, alcance ou senioridade em relacao a fonte.\n"
    "Fato copiado ou parafraseado da fonte, sem relacao nova entre fatos, conta como sustentado. "
    "Julgue apenas pelo texto das fontes citadas, nunca por conhecimento externo.\n\n"
    "Explique cada resposta em uma frase curta em justificativa. Devolva uma nota por chave, com a "
    "chave copiada sem alteracao."
)


class NotaFrase(CamelModel):
    chave: str = Field(description="A chave da frase, copiada sem alteracao")
    responde_requisito: bool
    requisito_citado: str
    relacao_sustentada: bool
    justificativa: str


class AvaliacaoJuiz(CamelModel):
    notas: list[NotaFrase]


@dataclass(frozen=True)
class FraseJulgada:
    chave: str
    texto: str
    fontes: tuple[tuple[str, str], ...]


def _bloco_frase(frase: FraseJulgada) -> str:
    fontes = "\n".join(f"    [{fonte_id}] {texto}" for fonte_id, texto in frase.fontes) or "    nenhuma"
    return f"- chave: {frase.chave}\n  frase: {frase.texto}\n  fontes citadas:\n{fontes}"


def montar_user(descricao_vaga: str, keywords: list[Keyword], frases: list[FraseJulgada]) -> str:
    termos = ", ".join(k.termo for k in keywords) or "nenhuma"
    return (
        f"Descricao da vaga:\n{descricao_vaga}\n\nKeywords da vaga: {termos}\n\n"
        "Frases para avaliar:\n" + "\n\n".join(_bloco_frase(frase) for frase in frases)
    )


def julgar(descricao_vaga: str, keywords: list[Keyword], frases: list[FraseJulgada]) -> dict[str, NotaFrase]:
    if not frases:
        return {}
    esperadas = {frase.chave for frase in frases}

    def validar(resposta: AvaliacaoJuiz) -> None:
        faltando = esperadas - {nota.chave for nota in resposta.notas}
        if faltando:
            raise ValidacaoSemantica("faltam notas para as chaves: " + ", ".join(sorted(faltando)))

    resposta = complete_model(
        SYSTEM,
        montar_user(descricao_vaga, keywords, frases),
        AvaliacaoJuiz,
        chamador=CHAMADOR,
        esforco=ESFORCO,
        validar=validar,
    )
    return {nota.chave: nota for nota in resposta.notas if nota.chave in esperadas}
