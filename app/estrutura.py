from .casamento import compactar, termo_presente
from .fontes import Fonte
from .orcamento import (
    BULLETS_TOTAL,
    COMPETENCIAS_MAX_CATEGORIAS,
    COMPETENCIAS_MAX_TERMOS,
    EXPERIENCIAS_SEM_CORTE,
    PISO_BULLETS,
    RESUMO_MAX_FRASES,
    bullets_maximos,
)
from .renderizador import experiencias_por_recencia, realizacoes
from .schemas import (
    CategoriaCompetencias,
    EstruturaCurriculo,
    ExperienciaEstruturada,
    FraseFonte,
    GenerateCvRequest,
    PerfilMestre,
    TermoFonte,
)

ROTULO_COMPETENCIAS = {"pt": "Principais", "en": "Core", "es": "Principales"}


def titulo_do_perfil(perfil: PerfilMestre) -> FraseFonte | None:
    for experiencia_id, experiencia in experiencias_por_recencia(perfil):
        if experiencia.cargo.strip():
            return FraseFonte(texto=experiencia.cargo.strip(), fontes=[experiencia_id])
    return None


def _total_bullets(experiencias: list[ExperienciaEstruturada]) -> int:
    return sum(len(item.bullets) for item in experiencias)


def aplicar_orcamento(estrutura: EstruturaCurriculo, perfil: PerfilMestre) -> EstruturaCurriculo:
    por_id = {item.experiencia_id: item.bullets for item in estrutura.experiencias}
    experiencias = [
        ExperienciaEstruturada(
            experiencia_id=experiencia_id,
            bullets=list(por_id.get(experiencia_id, []))[: bullets_maximos(posicao)],
        )
        for posicao, (experiencia_id, _) in enumerate(experiencias_por_recencia(perfil))
    ]
    while _total_bullets(experiencias) > BULLETS_TOTAL:
        alvo = next((item for item in reversed(experiencias) if len(item.bullets) > PISO_BULLETS), None)
        if alvo is None:
            break
        alvo.bullets.pop()
    competencias: list[CategoriaCompetencias] = []
    restantes = COMPETENCIAS_MAX_TERMOS
    for categoria in estrutura.competencias[:COMPETENCIAS_MAX_CATEGORIAS]:
        termos = categoria.termos[:restantes]
        restantes -= len(termos)
        if termos:
            competencias.append(CategoriaCompetencias(categoria=categoria.categoria, termos=termos))
    return EstruturaCurriculo(
        titulo=estrutura.titulo,
        resumo=estrutura.resumo[:RESUMO_MAX_FRASES],
        experiencias=experiencias,
        competencias=competencias,
        experiencias_omitidas=list(estrutura.experiencias_omitidas),
    )


def bullets_do_perfil(req: GenerateCvRequest, experiencia_id: str, limite: int) -> list[FraseFonte]:
    experiencia = dict(experiencias_por_recencia(req.perfil_mestre)).get(experiencia_id)
    if experiencia is None:
        return []
    itens = realizacoes(experiencia)
    termos = [k.termo for k in req.keywords if k.termo.strip()]
    pontuados = sorted(
        range(len(itens)),
        key=lambda i: (-sum(1 for termo in termos if termo_presente(termo, itens[i])), i),
    )
    escolhidos = sorted(pontuados[:limite])
    return [FraseFonte(texto=itens[i], fontes=[experiencia_id]) for i in escolhidos]


def _fonte_do_termo(termo: str, fontes: dict[str, Fonte]) -> str | None:
    return next((f.id for f in fontes.values() if f.factual and termo_presente(termo, f.texto)), None)


def competencias_do_perfil(
    req: GenerateCvRequest, fontes: dict[str, Fonte], idioma: str
) -> list[CategoriaCompetencias]:
    vistos: set[str] = set()
    termos: list[TermoFonte] = []
    candidatos = [k.termo.strip() for k in sorted(req.keywords, key=lambda k: -k.peso)]
    candidatos += [s.strip() for s in req.perfil_mestre.skills]
    for termo in candidatos:
        chave = compactar(termo)
        if not termo or chave in vistos:
            continue
        fonte_id = _fonte_do_termo(termo, fontes)
        if fonte_id:
            vistos.add(chave)
            termos.append(TermoFonte(termo=termo, fonte=fonte_id))
    if not termos:
        return []
    return [CategoriaCompetencias(categoria=ROTULO_COMPETENCIAS[idioma], termos=termos)]


def estrutura_do_perfil(req: GenerateCvRequest, fontes: dict[str, Fonte], idioma: str) -> EstruturaCurriculo:
    perfil = req.perfil_mestre
    resumo = [FraseFonte(texto=perfil.resumo.strip(), fontes=["resumo"])] if perfil.resumo.strip() else []
    experiencias = [
        ExperienciaEstruturada(
            experiencia_id=experiencia_id,
            bullets=bullets_do_perfil(req, experiencia_id, bullets_maximos(posicao)),
        )
        for posicao, (experiencia_id, _) in enumerate(experiencias_por_recencia(perfil))
    ]
    estrutura = EstruturaCurriculo(
        titulo=titulo_do_perfil(perfil),
        resumo=resumo,
        experiencias=experiencias,
        competencias=competencias_do_perfil(req, fontes, idioma),
    )
    return aplicar_orcamento(estrutura, perfil)


def _cortar_um_nivel(estrutura: EstruturaCurriculo, perfil: PerfilMestre) -> bool:
    omitidas = set(estrutura.experiencias_omitidas)
    por_id = {item.experiencia_id: item for item in estrutura.experiencias}
    visiveis = [
        experiencia_id for experiencia_id, _ in experiencias_por_recencia(perfil)
        if experiencia_id not in omitidas
    ]
    tamanhos = {i: len(por_id[i].bullets) if i in por_id else 0 for i in visiveis}
    maior = max(tamanhos.values(), default=0)
    if maior > PISO_BULLETS:
        for experiencia_id, tamanho in tamanhos.items():
            if tamanho == maior:
                por_id[experiencia_id].bullets.pop()
        return True
    if len(visiveis) > EXPERIENCIAS_SEM_CORTE:
        estrutura.experiencias_omitidas.append(visiveis[-1])
        return True
    return False


def cortar(estrutura: EstruturaCurriculo, perfil: PerfilMestre, nivel: int) -> EstruturaCurriculo:
    cortada = estrutura.model_copy(deep=True)
    for _ in range(nivel):
        if not _cortar_um_nivel(cortada, perfil):
            break
    return cortada
