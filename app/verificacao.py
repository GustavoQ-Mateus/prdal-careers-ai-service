import re
from collections.abc import Callable
from dataclasses import dataclass

from .casamento import compactar, dicionario_canonico, termo_presente
from .fontes import TIPO_EXPERIENCIA, Fonte, nota_factual
from .schemas import FraseFonte, GenerateCvRequest
from .text import normalize

UNIDADES = (
    "%", "x", "vezes", "pp", "p.p.", "ms", "s", "segundos", "minutos", "min", "h", "horas",
    "dias", "semanas", "meses", "anos", "mil", "k", "milhao", "milhoes", "milhão", "milhões",
    "bilhao", "bilhoes", "bilhão", "bilhões", "usuarios", "usuários", "clientes", "pessoas",
    "lojas", "modulos", "módulos", "projetos", "sistemas", "pacientes", "leitos", "alunos",
)
_UNIDADE = "|".join(sorted((re.escape(u) for u in UNIDADES), key=len, reverse=True))
METRICA_RE = re.compile(
    rf"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*({_UNIDADE})(?![\w%])",
    re.IGNORECASE,
)
MOEDA_RE = re.compile(r"(R\$|US\$|€|\$)\s*(\d+(?:[.,]\d+)*)", re.IGNORECASE)
VOCABULARIO_INTERNO = ("ferramenta por extenso", "resultado real", "verbo de acao")
_TOKEN_RE = re.compile(r"[^\s,;:()\[\]{}\"'!?]+")
_FIM_DE_FRASE = ".!?:"
_ANTES_DO_TOKEN = " \t\n(\"'["
ROTULO_RESULTADO_RE = re.compile(r"\bresultado\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class Rejeicao:
    chave: str
    texto: str
    fontes: tuple[str, ...]
    motivo: str


def termos_reconhecidos(req: GenerateCvRequest) -> list[str]:
    candidatos = [k.termo for k in req.keywords]
    candidatos += list(req.perfil_mestre.skills)
    candidatos += [forma for grupo in dicionario_canonico() for forma in grupo]
    vistos: set[str] = set()
    termos: list[str] = []
    for termo in candidatos:
        chave = compactar(termo)
        if termo.strip() and chave and chave not in vistos:
            vistos.add(chave)
            termos.append(termo.strip())
    return termos


def _numero(valor: str) -> str:
    return valor.replace(",", ".")


def _metricas_sem_fonte(texto: str, fonte: str) -> list[str]:
    faltando: list[str] = []
    fonte_norm = normalize(fonte)
    for match in METRICA_RE.finditer(texto):
        numero, unidade = _numero(match.group(1)), normalize(match.group(2))
        presente = any(
            _numero(m.group(1)) == numero and normalize(m.group(2)) == unidade
            for m in METRICA_RE.finditer(fonte_norm)
        )
        if not presente:
            faltando.append(match.group(0).strip())
    for match in MOEDA_RE.finditer(texto):
        numero = _numero(match.group(2))
        if not any(_numero(m.group(2)) == numero for m in MOEDA_RE.finditer(fonte)):
            faltando.append(match.group(0).strip())
    return faltando


def _vazamento(texto: str) -> list[str]:
    normalizado = normalize(texto)
    achados = [termo for termo in VOCABULARIO_INTERNO if termo in normalizado]
    if ROTULO_RESULTADO_RE.search(texto):
        achados.append("resultado:")
    return achados


def _inicio_de_frase(texto: str, posicao: int) -> bool:
    anterior = texto[:posicao].rstrip(_ANTES_DO_TOKEN)
    return not anterior or anterior[-1] in _FIM_DE_FRASE


def nomes_proprios(texto: str) -> list[str]:
    nomes: list[str] = []
    for match in _TOKEN_RE.finditer(texto):
        token = match.group().strip(".-/")
        letras = [c for c in token if c.isalpha()]
        simbolo = "#" in token or "+" in token
        if not letras or (len([c for c in token if c.isalnum()]) < 2 and not simbolo):
            continue
        maiuscula_no_meio = any(c.isupper() for c in token[1:])
        capitalizada = token[0].isupper() and not _inicio_de_frase(texto, match.start())
        tecnico = any(c.isdigit() for c in token) or simbolo
        if (maiuscula_no_meio or capitalizada or tecnico) and token not in nomes:
            nomes.append(token)
    return nomes


Permissao = Callable[[Fonte], str | None]


def permissao_do_bullet(experiencia_id: str) -> Permissao:
    def permitir(fonte: Fonte) -> str | None:
        if fonte.id == experiencia_id or nota_factual(fonte):
            return None
        if fonte.tipo == TIPO_EXPERIENCIA:
            return f"bullet cita outra experiencia: {fonte.id}"
        return f"bullet so pode citar a propria experiencia ou nota factual: {fonte.id}"

    return permitir


def motivo_da_rejeicao(
    frase: FraseFonte,
    fontes: dict[str, Fonte],
    termos: list[str],
    permissao: Permissao | None = None,
    nomes: bool = True,
) -> str | None:
    texto = frase.texto.strip()
    if not texto:
        return "frase vazia"
    citadas = [fonte_id.strip() for fonte_id in frase.fontes if fonte_id.strip()]
    if not citadas:
        return "frase sem fonte citada"
    for fonte_id in citadas:
        fonte = fontes.get(fonte_id)
        if fonte is None:
            return f"fonte inexistente: {fonte_id}"
        if not fonte.factual:
            return f"fonte de apoio nao sustenta fato: {fonte_id}"
        if permissao and (motivo := permissao(fonte)):
            return motivo
    texto_fontes = "\n".join(fontes[fonte_id].texto for fonte_id in citadas)
    sem_fonte = [t for t in termos if termo_presente(t, texto) and not termo_presente(t, texto_fontes)]
    if nomes:
        sem_fonte += [
            n for n in nomes_proprios(texto)
            if not termo_presente(n, texto_fontes) and not any(termo_presente(n, t) for t in sem_fonte)
        ]
    if sem_fonte:
        return "termo ausente das fontes citadas: " + ", ".join(sem_fonte)
    metricas = _metricas_sem_fonte(texto, texto_fontes)
    if metricas:
        return "numero ausente das fontes citadas: " + ", ".join(metricas)
    vazamento = _vazamento(texto)
    if vazamento:
        return "vocabulario interno da redacao no texto: " + ", ".join(vazamento)
    return None


def motivo_da_competencia(termo: str, fonte_id: str, fontes: dict[str, Fonte]) -> str | None:
    if not termo.strip():
        return "termo vazio"
    fonte = fontes.get(fonte_id.strip())
    if fonte is None:
        return f"fonte inexistente: {fonte_id}"
    if not fonte.factual:
        return f"fonte de apoio nao sustenta fato: {fonte_id}"
    if not termo_presente(termo, fonte.texto):
        return f"termo ausente da fonte citada: {termo}"
    return None
