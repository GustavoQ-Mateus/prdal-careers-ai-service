import re
import unicodedata

from .schemas import (
    Certificacao,
    EstruturaCurriculo,
    ExperienciaPerfil,
    Formacao,
    Local,
    PerfilMestre,
    Vaga,
)
from .secoes import cabecalhos
from .text import normalize

BULLET_RE = re.compile(r"^-\s+(.+)$")
LINHA_TECNOLOGIAS_RE = re.compile(r"^tecnologias\s*:", re.IGNORECASE)
PONTUACAO_ASCII = {
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-",
    0x2014: "-", 0x2015: "-", 0x2212: "-",
}
LINKS_CABECALHO = ("linkedin", "github", "site")
TERMO_ATUAL = {"pt": "atual", "en": "present", "es": "actual"}
STATUS_FORMACAO = {
    "pt": {"concluido": "concluído", "em_andamento": "em andamento", "trancado": "trancado"},
    "en": {"concluido": "completed", "em_andamento": "in progress", "trancado": "on hold"},
    "es": {"concluido": "concluido", "em_andamento": "en curso", "trancado": "interrumpido"},
}
MESES = {
    "jan": "01", "janeiro": "01", "january": "01", "enero": "01",
    "fev": "02", "fevereiro": "02", "feb": "02", "february": "02", "febrero": "02",
    "mar": "03", "marco": "03", "março": "03", "march": "03", "marzo": "03",
    "abr": "04", "abril": "04", "apr": "04", "april": "04",
    "mai": "05", "maio": "05", "may": "05", "mayo": "05",
    "jun": "06", "junho": "06", "june": "06", "junio": "06",
    "jul": "07", "julho": "07", "july": "07", "julio": "07",
    "ago": "08", "agosto": "08", "aug": "08", "august": "08",
    "set": "09", "setembro": "09", "sep": "09", "september": "09", "septiembre": "09",
    "out": "10", "outubro": "10", "oct": "10", "october": "10", "octubre": "10",
    "nov": "11", "novembro": "11", "november": "11", "noviembre": "11",
    "dez": "12", "dezembro": "12", "dec": "12", "december": "12", "diciembre": "12",
}


def _sem_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    ).lower()


def texto_limpo(texto: str) -> str:
    return re.sub(r"\s+", " ", texto.translate(PONTUACAO_ASCII)).strip()


def idioma_da_vaga(vaga: Vaga) -> str:
    texto = normalize(f"{vaga.titulo} {vaga.descricao}")
    en = ["requirements", "responsibilities", "experience", "english", "skills", "we are", "hiring"]
    es = [
        "espanol", "desarrollador", "desarrolladora", "conocimientos",
        "habilidades", "formacion", "trabajo remoto", "postulate",
    ]
    if sum(t in texto for t in en) >= 2:
        return "en"
    if "español" in vaga.descricao.lower() or sum(t in texto for t in es) >= 2:
        return "es"
    return "pt"


def _link(valor: str, url: str | None) -> str:
    return f"[{valor}]({url})" if url else valor


def texto_local(local: Local | None) -> str:
    if not local:
        return ""
    return " - ".join(parte for parte in (local.cidade.strip(), local.estado.strip()) if parte)


def linha_contato(perfil: PerfilMestre) -> str:
    partes: list[str] = []
    telefone = next((t for t in perfil.telefones if t.principal and t.numero.strip()), None)
    if telefone:
        partes.append(" ".join(p for p in (telefone.ddi.strip(), telefone.numero.strip()) if p))
    email = next((e for e in perfil.emails if e.principal and e.valor.strip()), None)
    if email:
        valor = email.valor.strip()
        partes.append(_link(valor, f"mailto:{valor}" if "@" in valor else None))
    local = texto_local(perfil.endereco)
    if local:
        partes.append(local)
    for tipo in LINKS_CABECALHO:
        for link in perfil.links:
            url = link.url.strip()
            if link.tipo == tipo and url:
                partes.append(_link(url, url if "://" in url else f"https://{url}"))
    return " | ".join(partes)


def realizacoes(experiencia: ExperienciaPerfil) -> list[str]:
    if experiencia.realizacoes:
        return [item.strip().lstrip("- ").strip() for item in experiencia.realizacoes if item.strip()]
    linhas = [
        linha.strip() for linha in experiencia.descricao.splitlines()
        if linha.strip() and not LINHA_TECNOLOGIAS_RE.match(linha.strip())
    ]
    bullets = [m.group(1).strip() for linha in linhas if (m := BULLET_RE.match(linha))]
    if bullets:
        return bullets
    return [" ".join(linhas)] if linhas else []


def _periodo_mm_aaaa(periodo: str, idioma: str) -> str:
    texto = periodo.strip()

    def substituir(match: re.Match[str]) -> str:
        mes = _sem_acentos(match.group(1)).rstrip(".")
        return f"{MESES.get(mes, match.group(1))}/{match.group(2)}"

    nomes = "|".join(sorted((re.escape(m) for m in MESES), key=len, reverse=True))
    texto = re.sub(rf"\b({nomes})\.?\s+(\d{{4}})\b", substituir, texto, flags=re.IGNORECASE)
    texto = re.sub(
        r"\s+(?:a|ate|até|to|hasta)\s+(?:atual|present|actual)\b",
        f" - {TERMO_ATUAL[idioma]}", texto, flags=re.IGNORECASE,
    )
    return re.sub(r"\s*[\u2013\u2014]\s*", " - ", texto)


def _mes_ano(mes: int | None, ano: int | None) -> str:
    if ano is None:
        return ""
    return f"{mes:02d}/{ano}" if mes else str(ano)


def periodo_experiencia(experiencia: ExperienciaPerfil, idioma: str) -> str:
    if experiencia.periodo_legado.strip():
        return _periodo_mm_aaaa(experiencia.periodo_legado, idioma)
    inicio = _mes_ano(experiencia.data_inicio_mes, experiencia.data_inicio_ano)
    fim = (
        TERMO_ATUAL[idioma]
        if experiencia.atual
        else _mes_ano(experiencia.data_fim_mes, experiencia.data_fim_ano)
    )
    return f"{inicio} - {fim}" if inicio and fim else inicio or fim


def local_experiencia(experiencia: ExperienciaPerfil) -> str:
    return texto_local(experiencia.local) or experiencia.local_legado.strip()


def _recencia(experiencia: ExperienciaPerfil) -> tuple[int, tuple[int, int], tuple[int, int]]:
    inicio = (experiencia.data_inicio_ano or 0, experiencia.data_inicio_mes or 0)
    if experiencia.atual:
        return (1, inicio, inicio)
    fim = (
        (experiencia.data_fim_ano, experiencia.data_fim_mes or 0)
        if experiencia.data_fim_ano
        else inicio
    )
    return (0, fim, inicio)


def ids_experiencias(perfil: PerfilMestre) -> list[str]:
    vistos: set[str] = set()
    ids: list[str] = []
    for indice, experiencia in enumerate(perfil.experiencias):
        candidato = experiencia.id.strip() or f"experiencia-{indice + 1}"
        if candidato in vistos:
            candidato = f"{candidato}-{indice + 1}"
        vistos.add(candidato)
        ids.append(candidato)
    return ids


def experiencias_por_recencia(perfil: PerfilMestre) -> list[tuple[str, ExperienciaPerfil]]:
    pares = list(zip(ids_experiencias(perfil), perfil.experiencias))
    return sorted(pares, key=lambda par: _recencia(par[1]), reverse=True)


def texto_formacao(formacao: Formacao, idioma: str) -> str:
    grau, curso = formacao.grau.strip(), formacao.curso.strip()
    conector = {"pt": "em", "en": "in", "es": "en"}[idioma]
    titulo = f"{grau} {conector} {curso}" if grau and curso else grau or curso
    inicio = _mes_ano(formacao.inicio_mes, formacao.inicio_ano)
    fim = _mes_ano(formacao.fim_mes, formacao.fim_ano)
    if not fim and formacao.status in ("", "em_andamento"):
        fim = TERMO_ATUAL[idioma] if inicio else ""
    periodo = f"{inicio} - {fim}" if inicio and fim else inicio or fim
    status = STATUS_FORMACAO[idioma].get(formacao.status, "")
    return " | ".join(p for p in (formacao.instituicao.strip(), titulo, periodo, status) if p)


def texto_certificacao(certificacao: Certificacao) -> str:
    return ", ".join(p for p in (certificacao.titulo.strip(), certificacao.descricao.strip()) if p)


def texto_experiencia(experiencia: ExperienciaPerfil) -> str:
    partes = [
        experiencia.cargo,
        experiencia.empresa,
        periodo_experiencia(experiencia, "pt"),
        local_experiencia(experiencia),
        experiencia.descricao,
    ]
    partes.extend(experiencia.realizacoes)
    return "\n".join(parte for parte in partes if parte)


def texto_perfil(perfil: PerfilMestre) -> str:
    partes = [
        perfil.nome, linha_contato(perfil), perfil.resumo,
        " ".join(texto_experiencia(e) for e in perfil.experiencias),
        " ".join(texto_formacao(f, "pt") for f in perfil.formacao),
        " ".join(texto_certificacao(c) for c in perfil.certificacoes),
        " ".join(perfil.idiomas), " ".join(perfil.skills),
    ]
    return "\n".join(p for p in partes if p)


def renderizar(perfil: PerfilMestre, estrutura: EstruturaCurriculo, idioma: str) -> str:
    h = cabecalhos(idioma)
    linhas = [f"# {texto_limpo(perfil.nome)}".strip()]
    if estrutura.titulo and estrutura.titulo.texto.strip():
        linhas.append(f"**{texto_limpo(estrutura.titulo.texto)}**")
    contato = linha_contato(perfil)
    if contato:
        linhas += ["", contato]

    resumo = " ".join(texto_limpo(frase.texto) for frase in estrutura.resumo if frase.texto.strip())
    _secao(linhas, h["resumo"], [resumo])

    competencias = []
    for categoria in estrutura.competencias:
        termos = [texto_limpo(t.termo) for t in categoria.termos if t.termo.strip()]
        if termos:
            competencias.append(f"- {texto_limpo(categoria.categoria)}: {', '.join(termos)}")
    _secao(linhas, h["competencias"], competencias)

    bullets_por_id = {item.experiencia_id: item.bullets for item in estrutura.experiencias}
    omitidas = set(estrutura.experiencias_omitidas)
    experiencias: list[str] = []
    for experiencia_id, experiencia in experiencias_por_recencia(perfil):
        if experiencia_id in omitidas:
            continue
        empresa = texto_limpo(experiencia.empresa)
        cargo = texto_limpo(experiencia.cargo)
        periodo = periodo_experiencia(experiencia, idioma)
        cabecalho = " | ".join(p for p in (f"**{empresa}**" if empresa else "", cargo, periodo) if p)
        experiencias += ["", cabecalho]
        for bullet in bullets_por_id.get(experiencia_id, []):
            if bullet.texto.strip():
                experiencias.append(f"- {texto_limpo(bullet.texto)}")
    _secao(linhas, h["experiencia"], experiencias)

    formacoes = [texto for f in perfil.formacao if (texto := texto_formacao(f, idioma))]
    certificacoes = [texto for c in perfil.certificacoes if (texto := texto_certificacao(c))]
    idiomas = [item.strip() for item in perfil.idiomas if item.strip()]
    _secao(linhas, h["formacao"], formacoes)
    _secao(linhas, h["certificacoes"], [f"- {item}" for item in certificacoes])
    _secao(linhas, h["idiomas"], [" | ".join(idiomas)])
    return "\n".join(linhas).strip()


def _secao(linhas: list[str], titulo: str, corpo: list[str]) -> None:
    if any(linha.strip() for linha in corpo):
        linhas += ["", f"## {titulo}", *corpo]
