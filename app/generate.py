import re
from typing import Any

from .llm import LLMUnavailable, complete_model
from .schemas import (
    AtsAnalysis,
    ExperienciaPerfil,
    GenerateCvRequest,
    GenerateCvResponse,
    GeneratePipelineResponse,
    PerfilMestre,
)
from .score import calcular_score
from .text import normalize

SYSTEM = (
    "Voce reescreve curriculos ATS em Markdown. Use somente fatos presentes no "
    "perfil-mestre e no contexto RAG fornecidos. Nunca invente metricas, datas, "
    "empresas, contatos, tecnologias ou autoria individual. Responda em JSON."
)


def _user(req: GenerateCvRequest) -> str:
    termos = ", ".join(k.termo for k in req.keywords)
    return (
        "Gere o Markdown do curriculo e devolva JSON no formato "
        '{"markdown":"..."}.\n\n'
        "Regras obrigatorias:\n"
        "- Processar uma vaga por vez e escrever no idioma da vaga.\n"
        "- Estrutura limpa: nome e titulo, contato, resumo profissional, "
        "competencias, experiencia profissional, formacao, certificacoes e idiomas.\n"
        "- Sem tabelas, colunas, icones, travessao, parenteses decorativos ou "
        "keyword stuffing.\n"
        "- Datas em MM/AAAA quando existirem no perfil.\n"
        "- Titulo alinhado a vaga e espelhamento honesto dos termos exatos da vaga.\n"
        "- Cada bullet inicia com verbo de acao e combina contexto, tecnologia e "
        "impacto real.\n"
        "- Nao misture experiencias, nao transforme contribuicao de time em autoria "
        "individual e nao invente fatos ausentes.\n\n"
        f"Perfil-mestre:\n{req.perfil_mestre.model_dump_json()}\n\n"
        f"Vaga: {req.vaga.titulo} @ {req.vaga.empresa}\n"
        f"Descricao da vaga:\n{req.vaga.descricao}\n\n"
        f"Palavras-chave a priorizar: {termos}\n\n"
        f"Contexto adicional:\n{chr(10).join(req.contexto)}"
    )


def _linha_contato(contato: dict[str, Any]) -> str:
    partes = [str(v) for v in contato.values() if v]
    return " | ".join(partes)


def _texto_perfil(perfil: PerfilMestre) -> str:
    partes = [
        perfil.nome,
        str(perfil.contato),
        perfil.resumo,
        " ".join(_texto_experiencia(e) for e in perfil.experiencias),
        " ".join(perfil.formacao),
        " ".join(perfil.skills),
    ]
    return "\n".join(p for p in partes if p)


def _idioma(req: GenerateCvRequest) -> str:
    texto = normalize(f"{req.vaga.titulo} {req.vaga.descricao}")
    en = [
        "requirements",
        "responsibilities",
        "experience",
        "english",
        "skills",
        "we are",
        "hiring",
        "developer",
        "build",
        "cloud",
        "deployment",
    ]
    es = ["requisitos", "responsabilidades", "experiencia", "espanol", "habilidades"]
    if sum(t in texto for t in en) >= 2:
        return "en"
    if "español" in req.vaga.descricao.lower() or sum(t in texto for t in es) >= 3:
        return "es"
    return "pt"


def _cabecalhos(idioma: str) -> dict[str, str]:
    if idioma == "en":
        return {
            "resumo": "PROFESSIONAL SUMMARY",
            "competencias": "SKILLS",
            "experiencia": "PROFESSIONAL EXPERIENCE",
            "formacao": "EDUCATION",
            "certificacoes": "CERTIFICATIONS",
            "idiomas": "LANGUAGES",
        }
    if idioma == "es":
        return {
            "resumo": "RESUMEN PROFESIONAL",
            "competencias": "COMPETENCIAS",
            "experiencia": "EXPERIENCIA PROFESIONAL",
            "formacao": "FORMACION ACADEMICA",
            "certificacoes": "CERTIFICACIONES",
            "idiomas": "IDIOMAS",
        }
    return {
        "resumo": "RESUMO PROFISSIONAL",
        "competencias": "COMPETENCIAS",
        "experiencia": "EXPERIENCIA PROFISSIONAL",
        "formacao": "FORMACAO ACADEMICA",
        "certificacoes": "CERTIFICACOES",
        "idiomas": "IDIOMAS",
    }


def _texto_experiencia(experiencia: ExperienciaPerfil) -> str:
    if experiencia.texto:
        return experiencia.texto
    partes = [
        f"Cargo: {experiencia.cargo}" if experiencia.cargo else "",
        f"Empresa: {experiencia.empresa}" if experiencia.empresa else "",
        f"Período: {experiencia.periodo}" if experiencia.periodo else "",
        f"Local: {experiencia.local}" if experiencia.local else "",
        f"Descrição: {experiencia.descricao}" if experiencia.descricao else "",
        f"Tecnologias e competências: {', '.join(experiencia.tecnologias)}"
        if experiencia.tecnologias
        else "",
    ]
    return "\n".join(parte for parte in partes if parte)


def _titulo_experiencia(experiencia: ExperienciaPerfil) -> str:
    if experiencia.cargo and experiencia.empresa:
        return f"{experiencia.cargo} na {experiencia.empresa}"
    return experiencia.cargo or experiencia.empresa or "Experiência"


def _deterministic(perfil: PerfilMestre) -> str:
    h = _cabecalhos("pt")
    titulo = perfil.skills[0] if perfil.skills else "Profissional"
    linhas = [f"# {perfil.nome} | {titulo}".strip()]
    contato = _linha_contato(perfil.contato)
    if contato:
        linhas.append(contato)
    if perfil.resumo:
        linhas += ["", f"## {h['resumo']}", perfil.resumo]
    if perfil.skills:
        linhas += ["", f"## {h['competencias']}", ", ".join(perfil.skills)]
    if perfil.experiencias:
        linhas += ["", f"## {h['experiencia']}"]
        for experiencia in perfil.experiencias:
            texto = _texto_experiencia(experiencia)
            if texto:
                periodo = experiencia.periodo or "periodo nao informado"
                empresa = experiencia.empresa or "Empresa"
                cargo = experiencia.cargo or "Cargo"
                linhas += ["", f"**{empresa}** | {cargo} | {periodo}", texto]
    if perfil.formacao:
        linhas += ["", f"## {h['formacao']}", *[f"- {formacao}" for formacao in perfil.formacao]]
    linhas += ["", f"## {h['certificacoes']}"]
    linhas += ["", f"## {h['idiomas']}"]
    return "\n".join(linhas).strip() or "# Curriculo"


def _competencias_relevantes(req: GenerateCvRequest) -> list[str]:
    perfil_texto = normalize(_texto_perfil(req.perfil_mestre))
    relevantes = [
        k.termo.strip()
        for k in req.keywords
        if k.termo.strip() and normalize(k.termo) in perfil_texto
    ]
    for skill in req.perfil_mestre.skills:
        if skill not in relevantes:
            relevantes.append(skill)
    return relevantes[:18]


def _deterministic_request(req: GenerateCvRequest) -> str:
    perfil = req.perfil_mestre
    idioma = _idioma(req)
    h = _cabecalhos(idioma)
    titulo = req.vaga.titulo or (perfil.skills[0] if perfil.skills else "Profissional")
    linhas = [f"# {perfil.nome} | {titulo}".strip()]
    contato = _linha_contato(perfil.contato)
    if contato:
        linhas.append(contato)

    competencias = _competencias_relevantes(req)
    resumo_kw = ", ".join(competencias[:6])
    if idioma == "en":
        resumo = (
            f"{perfil.resumo} Experience aligned with {req.vaga.titulo}, with factual "
            f"background in {resumo_kw}."
        ).strip()
    elif idioma == "es":
        resumo = (
            f"{perfil.resumo} Experiencia alineada con {req.vaga.titulo}, con base "
            f"factual en {resumo_kw}."
        ).strip()
    else:
        resumo = (
            f"{perfil.resumo} Experiencia alinhada a {req.vaga.titulo}, com base "
            f"factual em {resumo_kw}."
        ).strip()
    linhas += ["", f"## {h['resumo']}", resumo]
    linhas += ["", f"## {h['competencias']}", ", ".join(competencias)]

    if perfil.experiencias:
        linhas += ["", f"## {h['experiencia']}"]
        for experiencia in perfil.experiencias:
            empresa = experiencia.empresa or "Empresa"
            cargo = experiencia.cargo or "Cargo"
            periodo = experiencia.periodo or "periodo nao informado"
            tecnologias = ", ".join(experiencia.tecnologias[:8])
            descricao = experiencia.descricao.strip()
            linhas += ["", f"**{empresa}** | {cargo} | {periodo}"]
            if idioma == "en":
                linhas.append(
                    f"- Contributed with the team to {descricao} using {tecnologias}."
                )
            elif idioma == "es":
                linhas.append(
                    f"- Contribui junto al equipo en {descricao} usando {tecnologias}."
                )
            else:
                linhas.append(
                    f"- Contribui junto ao time em {descricao} usando {tecnologias}."
                )

    if perfil.formacao:
        linhas += ["", f"## {h['formacao']}", *[f"- {formacao}" for formacao in perfil.formacao]]
    linhas += ["", f"## {h['certificacoes']}", "", f"## {h['idiomas']}"]
    return "\n".join(linhas).strip()


def _limpar_markdown(markdown: str, req: GenerateCvRequest) -> str:
    texto = markdown.replace("—", "-").replace("–", "-")
    texto = re.sub(r"[ \t]+", " ", texto)
    idioma = _idioma(req)
    h = _cabecalhos(idioma)
    if "## " not in texto:
        texto = _deterministic(req.perfil_mestre)
    for secao in h.values():
        if f"## {secao}" not in texto.upper():
            texto += f"\n\n## {secao}\n"
    return texto.strip()


def _analise(markdown: str, req: GenerateCvRequest) -> AtsAnalysis:
    score = calcular_score(markdown, req.keywords)
    texto = normalize(markdown)
    encontradas = []
    ausentes = []
    for k in req.keywords:
        termo = k.termo.strip()
        if not termo:
            continue
        if normalize(termo) in texto:
            encontradas.append(termo)
        elif k.peso >= 0.6 or len(ausentes) < 8:
            ausentes.append(termo)

    pontos = []
    if score.breakdown.secoes < 100:
        pontos.append("secoes ATS obrigatorias incompletas")
    if score.breakdown.keyword_match < 35:
        pontos.append("baixa aderencia as keywords criticas da vaga")
    if not re.search(r"\d{2}/\d{4}", markdown):
        pontos.append("datas em MM/AAAA ausentes ou insuficientes")
    veredicto = (
        "Passa no filtro automatico com ajustes finos recomendados."
        if score.score >= 70 and not pontos
        else "Passa com risco no filtro automatico; revisar ausencias criticas."
        if score.score >= 55
        else "Eliminado no filtro automatico por baixa aderencia deterministica."
    )
    return AtsAnalysis(
        score=score.score,
        keywords_encontradas=encontradas,
        keywords_criticas_ausentes=ausentes,
        pontos_eliminatorios=pontos,
        veredicto=veredicto,
        breakdown=score.breakdown.model_dump(by_alias=True),
    )


def generate_cv(req: GenerateCvRequest) -> str:
    return generate_cv_pipeline(req).markdown


def generate_cv_pipeline(req: GenerateCvRequest) -> GeneratePipelineResponse:
    base = _deterministic_request(req)
    inicial = _analise(_texto_perfil(req.perfil_mestre), req)
    degradacao = None
    try:
        res = complete_model(SYSTEM, _user(req), GenerateCvResponse)
        if res.markdown.strip():
            markdown = _limpar_markdown(res.markdown, req)
        else:
            markdown = base
            degradacao = "Groq retornou resposta vazia; usado fallback factual."
    except LLMUnavailable as exc:
        markdown = base
        degradacao = f"Groq indisponivel; usado fallback factual: {exc}"
    final = _analise(markdown, req)
    return GeneratePipelineResponse(
        markdown=markdown,
        analise_inicial=inicial,
        analise_final=final,
        degradacao=degradacao,
    )
