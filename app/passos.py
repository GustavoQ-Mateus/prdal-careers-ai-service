from dataclasses import asdict

from .degradacao import KEYWORDS_INDISPONIVEIS, registrar
from .generate import (
    _prompt_version,
    montar_geracao,
    rascunho_geracao,
    reparar_geracao,
    verificar_geracao,
)
from .keywords import extract_keywords
from .llm import LLMUnavailable, TetoDeRequisicoes, operacao, teto_de_requisicoes
from .schemas import (
    GenerateCvRequest,
    KeywordsRequest,
    KeywordsResponse,
    MontarGeracaoRequest,
    RascunhoGeracaoResponse,
    RepararGeracaoRequest,
    VerificarGeracaoRequest,
    UsoLlm,
)


def executar(passo: str, dados: dict, prazo_ms: int | None = None, operacao_id: str | None = None) -> dict:
    with operacao(prazo_ms, operacao_id) as op:
        if passo == "rascunho":
            try:
                with teto_de_requisicoes(1):
                    resposta = RascunhoGeracaoResponse(
                        rascunho=rascunho_geracao(GenerateCvRequest.model_validate(dados)),
                        prompt_version=_prompt_version(),
                    )
            except TetoDeRequisicoes as exc:
                raise LLMUnavailable("resposta invalida apos reparo") from exc
        elif passo == "verificar":
            resposta = verificar_geracao(VerificarGeracaoRequest.model_validate(dados))
        elif passo == "reparar":
            resposta = reparar_geracao(RepararGeracaoRequest.model_validate(dados))
        elif passo == "montar":
            resposta = montar_geracao(MontarGeracaoRequest.model_validate(dados))
        elif passo == "keywords":
            req = KeywordsRequest.model_validate(dados)
            try:
                resposta = KeywordsResponse(keywords=extract_keywords(req.descricao))
            except LLMUnavailable as exc:
                resposta = KeywordsResponse(
                    keywords=[], status="PENDENTE",
                    degradacao=registrar(KEYWORDS_INDISPONIVEIS, exc),
                )
        else:
            raise ValueError(f"passo desconhecido: {passo}")
        resposta.uso = UsoLlm(**asdict(op.uso))
        resposta.modelo = op.modelo
        if hasattr(resposta, "requisicoes"):
            resposta.requisicoes = op.requisicoes
        return resposta.model_dump(by_alias=True)
