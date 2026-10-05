import re
import unicodedata

from .llm import LLMUnavailable, ValidacaoSemantica, complete_model, modelo_configurado, montar_requisicao, MAX_TOKENS_PADRAO, _env_int
from .schemas import Keyword, KeywordsLlmResponse

ESFORCO = "low"

TIPOS_VALIDOS = {"stack", "ferramenta", "metodologia", "dominio_negocio", "certificacao"}

TERMOS_FORA_DAS_CATEGORIAS = {
    "brasil", "portugal", "reino unido", "diversidade", "respeito", "etica",
    "inovacao", "comunidade tech", "expansao global",
}

SYSTEM = (
    "Voce extrai palavras-chave relevantes de descricoes de vaga para "
    "otimizacao ATS. Cada termo deve ser uma competencia tecnica, linguagem, "
    "framework, ferramenta, metodologia, certificacao ou dominio de negocio "
    "explicito em um requisito da vaga. Nome de pais, adjetivo institucional "
    "(como diversidade, respeito, etica ou inovacao), valor corporativo e frase "
    "de missao/employer branding nunca sao keywords validas. Classifique cada "
    "termo em stack, ferramenta, metodologia, dominio_negocio ou certificacao. "
    "Cada palavra-chave tem um peso de 0 a 1 conforme a importancia. Responda em JSON.\n\n"
    "Extraia ate 15 palavras-chave da descricao recebida e devolva JSON no "
    'formato {"keywords":[{"termo":"...","peso":0.0,"tipo":"stack"}]}. '
    "Inclua somente termos tecnicos das cinco categorias permitidas. "
    "Nao inclua paises, valores institucionais, adjetivos de cultura ou frases de missao."
)


def _user(descricao: str) -> str:
    return f"Descricao:\n{descricao}"


def _normalizar(texto: str) -> str:
    sem_acento = "".join(
        caractere
        for caractere in unicodedata.normalize("NFD", texto.lower())
        if not unicodedata.combining(caractere)
    )
    return re.sub(r"\s+", " ", sem_acento).strip()


def _validar_keyword(keyword) -> bool:
    termo = _normalizar(keyword.termo)
    if not termo or keyword.tipo not in TIPOS_VALIDOS:
        return False
    if termo in TERMOS_FORA_DAS_CATEGORIAS:
        return False
    if len(termo.split()) > 6:
        return False
    return True


def _exigir_keyword_valida(res: KeywordsLlmResponse) -> None:
    if not any(_validar_keyword(item) for item in res.keywords):
        raise ValidacaoSemantica("nenhuma palavra-chave tecnica valida nas cinco categorias permitidas")


def extract_keywords(descricao: str) -> list[Keyword]:
    res = complete_model(
        SYSTEM,
        _user(descricao),
        KeywordsLlmResponse,
        chamador="keywords",
        esforco=ESFORCO,
        validar=_exigir_keyword_valida,
    )
    keywords = [
        Keyword(termo=item.termo.strip(), peso=item.peso, tipo=item.tipo)
        for item in res.keywords
        if _validar_keyword(item)
    ]
    if not keywords:
        raise LLMUnavailable("extracao de keywords retornou lista vazia")
    return keywords


def preparar_lote(descricao: str) -> dict:
    modelo = modelo_configurado()
    if not modelo:
        raise LLMUnavailable("AI_MODEL ausente")
    return montar_requisicao(modelo, SYSTEM, _user(descricao), KeywordsLlmResponse, ESFORCO, _env_int("AI_MAX_TOKENS", MAX_TOKENS_PADRAO))


def interpretar_lote(resposta: dict) -> list[Keyword]:
    if resposta.get("stop_reason") != "end_turn":
        raise ValueError("resposta de keywords incompleta")
    texto = "".join(bloco.get("text", "") for bloco in resposta.get("content", []) if bloco.get("type") == "text")
    resultado = KeywordsLlmResponse.model_validate_json(texto)
    _exigir_keyword_valida(resultado)
    return [
        Keyword(termo=item.termo.strip(), peso=item.peso, tipo=item.tipo)
        for item in resultado.keywords
        if _validar_keyword(item)
    ]
