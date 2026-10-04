import re

from .casamento import normalizar

CABECALHOS = {
    "pt": {
        "resumo": "RESUMO PROFISSIONAL", "competencias": "COMPETÊNCIAS",
        "experiencia": "EXPERIÊNCIA PROFISSIONAL", "formacao": "FORMAÇÃO ACADÊMICA",
        "certificacoes": "CERTIFICAÇÕES", "idiomas": "IDIOMAS",
    },
    "en": {
        "resumo": "PROFESSIONAL SUMMARY", "competencias": "SKILLS",
        "experiencia": "PROFESSIONAL EXPERIENCE", "formacao": "EDUCATION",
        "certificacoes": "CERTIFICATIONS", "idiomas": "LANGUAGES",
    },
    "es": {
        "resumo": "RESUMEN PROFESIONAL", "competencias": "COMPETENCIAS",
        "experiencia": "EXPERIENCIA PROFESIONAL", "formacao": "FORMACIÓN ACADÉMICA",
        "certificacoes": "CERTIFICACIONES", "idiomas": "IDIOMAS",
    },
}

SECOES_OBRIGATORIAS = ("resumo", "competencias", "experiencia", "formacao")

_HEADING_RE = re.compile(r"^##(?!#)[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def _chave_heading(titulo: str) -> str:
    return re.sub(r"\s+", " ", normalizar(titulo).strip(" *:#")).strip()


_SECAO_POR_HEADING = {
    _chave_heading(titulo): secao
    for por_idioma in CABECALHOS.values()
    for secao, titulo in por_idioma.items()
}


def cabecalhos(idioma: str) -> dict[str, str]:
    return dict(CABECALHOS.get(idioma, CABECALHOS["pt"]))


def secao_do_heading(titulo: str) -> str | None:
    return _SECAO_POR_HEADING.get(_chave_heading(titulo))


def secoes_reconhecidas(markdown: str) -> dict[str, str]:
    headings = list(_HEADING_RE.finditer(markdown))
    secoes: dict[str, str] = {}
    for i, heading in enumerate(headings):
        secao = secao_do_heading(heading.group(1))
        if not secao:
            continue
        fim = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        corpo = markdown[heading.end():fim].strip()
        secoes[secao] = f"{secoes[secao]}\n{corpo}".strip() if secao in secoes else corpo
    return {secao: corpo for secao, corpo in secoes.items() if corpo}
