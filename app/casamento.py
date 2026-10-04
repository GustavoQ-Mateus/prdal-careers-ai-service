import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ARQUIVO_SINONIMOS = Path(__file__).resolve().parent / "taxonomia" / "sinonimos.v1.json"

_ANTES = r"(?<![a-z0-9])(?<![a-z0-9]\.)"
_DEPOIS = r"(?![a-z0-9#+$])(?!\.[a-z0-9])"
_VERSAO_APOS_SIMBOLO = r"(?:\d+)?"
_SEPARADOR_COMPOSTO = r"[ \t-]+"
_PALAVRA_TECNICA = re.compile(r"[a-z0-9#+]+(?:[./-][a-z0-9#+]+)*")
_ESPACO_ENTRE_PALAVRAS = re.compile(r"[ \t-]+")
_JANELA_COMPACTA = 3
_MINIMO_COMPACTO = 3


@dataclass(frozen=True)
class Ocorrencia:
    termo: str
    forma: str
    inicio: int
    fim: int


def _base(caractere: str) -> str:
    return unicodedata.normalize("NFD", caractere)[0].lower()[0]


def normalizar(texto: str) -> str:
    return "".join(_base(c) for c in texto)


def compactar(texto: str) -> str:
    return re.sub(r"[^a-z0-9#+]+", "", normalizar(texto))


@lru_cache(maxsize=1)
def _grupos() -> tuple[tuple[str, ...], ...]:
    dados = json.loads(ARQUIVO_SINONIMOS.read_text(encoding="utf-8"))
    return tuple(tuple(grupo) for grupo in dados["grupos"])


def dicionario_canonico() -> tuple[tuple[str, ...], ...]:
    return _grupos()


@lru_cache(maxsize=1)
def _grupo_por_forma() -> dict[str, tuple[str, ...]]:
    return {compactar(forma): grupo for grupo in _grupos() for forma in grupo}


def canonico(termo: str) -> str:
    grupo = _grupo_por_forma().get(compactar(termo))
    return grupo[0] if grupo else termo.strip()


def formas(termo: str) -> list[str]:
    limpo = termo.strip()
    grupo = _grupo_por_forma().get(compactar(limpo), ())
    return [limpo, *(forma for forma in grupo if compactar(forma) != compactar(limpo))]


def _casar_literal(forma: str, alvo: str) -> tuple[int, int] | None:
    partes = normalizar(forma).split()
    if not partes:
        return None
    antes = "" if partes[0].startswith(".") else _ANTES
    versao = _VERSAO_APOS_SIMBOLO if partes[-1].endswith(("+", "#")) else ""
    padrao = antes + _SEPARADOR_COMPOSTO.join(re.escape(p) for p in partes) + versao + _DEPOIS
    achado = re.search(padrao, alvo)
    return achado.span() if achado else None


def _casar_compacto(forma: str, alvo: str) -> tuple[int, int] | None:
    procurado = compactar(forma)
    if len(procurado) < _MINIMO_COMPACTO:
        return None
    palavras = list(_PALAVRA_TECNICA.finditer(alvo))
    for i, primeira in enumerate(palavras):
        acumulado = compactar(primeira.group())
        if acumulado == procurado:
            return primeira.span()
        for j in range(i + 1, min(i + _JANELA_COMPACTA, len(palavras))):
            entre = alvo[palavras[j - 1].end():palavras[j].start()]
            if not _ESPACO_ENTRE_PALAVRAS.fullmatch(entre):
                break
            acumulado += compactar(palavras[j].group())
            if acumulado == procurado:
                return primeira.start(), palavras[j].end()
            if not procurado.startswith(acumulado):
                break
    return None


def casar_termo(termo: str, texto: str) -> Ocorrencia | None:
    if not termo or not termo.strip() or not texto:
        return None
    alvo = normalizar(texto)
    for forma in formas(termo):
        intervalo = _casar_literal(forma, alvo) or _casar_compacto(forma, alvo)
        if intervalo:
            return Ocorrencia(termo=termo.strip(), forma=forma, inicio=intervalo[0], fim=intervalo[1])
    return None


def termo_presente(termo: str, texto: str) -> bool:
    return casar_termo(termo, texto) is not None
