from dataclasses import dataclass

from .renderizador import (
    experiencias_por_recencia,
    texto_certificacao,
    texto_experiencia,
    texto_formacao,
)
from .schemas import GenerateCvRequest

TIPO_EXPERIENCIA = "experiencia"
TIPO_NOTA = "nota"


@dataclass(frozen=True)
class Fonte:
    id: str
    tipo: str
    factual: bool
    titulo: str
    texto: str


def fontes_da_geracao(req: GenerateCvRequest) -> dict[str, Fonte]:
    perfil = req.perfil_mestre
    fontes: dict[str, Fonte] = {}

    def incluir(fonte: Fonte) -> None:
        if fonte.texto.strip() and fonte.id not in fontes:
            fontes[fonte.id] = fonte

    incluir(Fonte("resumo", "resumo", True, "Resumo", perfil.resumo.strip()))
    for experiencia_id, experiencia in experiencias_por_recencia(perfil):
        titulo = " na ".join(p for p in (experiencia.cargo.strip(), experiencia.empresa.strip()) if p)
        incluir(Fonte(experiencia_id, TIPO_EXPERIENCIA, True, titulo, texto_experiencia(experiencia)))
    incluir(Fonte("skills", "skills", True, "Skills", ", ".join(s.strip() for s in perfil.skills if s.strip())))
    for indice, formacao in enumerate(perfil.formacao):
        incluir(Fonte(f"formacao-{indice}", "formacao", True, "Formacao", texto_formacao(formacao, "pt")))
    for indice, certificacao in enumerate(perfil.certificacoes):
        incluir(Fonte(f"certificacao-{indice}", "certificacao", True, "Certificacao", texto_certificacao(certificacao)))
    incluir(Fonte("idiomas", "idiomas", True, "Idiomas", ", ".join(i.strip() for i in perfil.idiomas if i.strip())))
    for item in req.contexto:
        if item.tipo in (TIPO_NOTA, "candidatura"):
            incluir(Fonte(item.id, item.tipo, item.tipo == TIPO_NOTA and item.factual, item.titulo, item.texto.strip()))
    return fontes


def nota_factual(fonte: Fonte) -> bool:
    return fonte.tipo == TIPO_NOTA and fonte.factual
