import logging
import re
import unicodedata
from pathlib import Path
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
from .text import content_tokens, normalize

JOB_HEADER_RE = re.compile(r"^\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")
SKILL_RE = re.compile(r"^-\s*([^:]+):\s*(.+)$")
BULLET_RE = re.compile(r"^-\s+(.+)$")
FORMULA_LABEL_RE = re.compile(r"\bresultado\s*:", re.IGNORECASE)
FORMULA_LEAK_TERMS = ("ferramenta por extenso", "resultado real", "verbo de acao")
PONTUACAO_ASCII = {
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-",
    0x2014: "-", 0x2015: "-", 0x2212: "-",
}


class KeywordsUnavailable(ValueError):
    pass


def _repo_root() -> Path:
    atual = Path(__file__).resolve()
    for path in [Path.cwd(), *atual.parents]:
        if (path / ".claude" / "agents").exists():
            return path
    return Path.cwd()


REPO_ROOT = _repo_root()
SPEC_CANDIDATES = [
    Path.cwd() / ".claude" / "agents" / "modo-pipeline-curriculo.md",
    REPO_ROOT / ".claude" / "agents" / "modo-pipeline-curriculo.md",
]

TECH_CATALOG = (
    "django", "next.js", "nextjs", "kafka", "laravel", "vue", "ruby", "rails",
    "php", "go", "golang", "flask", "graphql", "kotlin", "swift", "oracle",
    "elasticsearch", "fastapi", "spring boot", "nestjs", "react", "angular",
    "postgresql", "mysql", "mongodb", "redis", "aws", "azure", "gcp",
    "terraform", "docker", "kubernetes", "c#", ".net", "rag", "llm", "yolo",
    "opencv", "celery", "sqs", "lambda", "aurora", "cloudfront", "whatsapp",
    "ocr", "paddleocr", "tesseract", "three.js", "leaflet", "vite",
)


def _read_spec() -> str:
    for path in SPEC_CANDIDATES:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return (
        "Modo Pipeline de Curriculo: Etapa 1 analise ATS; Etapa 2 reescrita "
        "otimizada; Etapa 3 score pos-geracao. Manter apenas fatos verdadeiros."
    )


def _system_prompt() -> str:
    return (
        "Voce e o motor do modo-pipeline-curriculo. Responda somente em JSON.\n\n"
        "Metodologia versionada do modo:\n"
        f"{_read_spec()}\n\n"
        "Regra de ouro multiusuario: use somente fatos presentes no perfil-mestre "
        "e no contexto factual deste request. Uma tecnologia da vaga so pode "
        "entrar no CV se existir nesses dados do usuario autenticado."
    )


def _json_model(model: Any) -> str:
    return model.model_dump_json(by_alias=True) if hasattr(model, "model_dump_json") else str(model)


def _user(
    req: GenerateCvRequest,
    analise_inicial: AtsAnalysis,
    lacunas: list[str],
    markdown_atual: str | None = None,
    erros: list[str] | None = None,
) -> str:
    termos = ", ".join(k.termo for k in req.keywords)
    lacunas_texto = ", ".join(lacunas) if lacunas else "nenhuma lacuna factual autorizada"
    reparos = "\n".join(f"- {erro}" for erro in (erros or [])) or "- nenhum"
    titulo_seguro = _titulo_vaga_seguro(req.vaga.titulo, req)
    h = _cabecalhos(_idioma(req))
    secoes_obrigatorias = " -> ".join(f"## {secao}" for secao in h.values())
    return (
        "Gere um curriculo tailored e devolva JSON exatamente no formato "
        '{"markdown":"..."}.\n\n'
        "Etapa 1 ja executada. Use a analise inicial abaixo como entrada da "
        "reescrita, nao a descarte:\n"
        f"{_json_model(analise_inicial)}\n\n"
        "Lacunas criticas que devem ser fechadas se couberem organicamente, usando "
        "os termos exatos da vaga e somente quando sustentadas pela fonte factual:\n"
        f"{lacunas_texto}\n"
        "Meta: overlap de vocabulario com a vaga >= 60%, sem keyword stuffing e "
        "sem secao 'palavras-chave'.\n\n"
        "Tailoring real: cada frase do resumo cita fato concreto, proibido RH "
        "generico sem fato. Abra cada experiencia pelo bullet mais aderente a "
        "vaga, sem mover tecnologia para experiencia onde nao foi usada nem "
        "apagar o bullet de maior responsabilidade tecnica ja registrada.\n\n"
        "Contrato obrigatorio do Markdown:\n"
        "- Primeira linha: '# NOME'. Segunda linha util: '**Titulo profissional**'. "
        "Depois, uma unica linha de contato no corpo. Nome e titulo nunca ficam na "
        "mesma linha.\n"
        f"- Use como titulo profissional '{titulo_seguro}', sem nome da empresa.\n"
        "- Secoes exatamente nesta ordem e com estes titulos literais: "
        f"{secoes_obrigatorias}. Nao traduza, nao use sinonimos e nao troque "
        "para espanhol/ingles quando a vaga estiver em portugues.\n"
        "- Competencias logo apos o resumo, em linhas '- Categoria: valor, valor'.\n"
        "- Cada experiencia usa '**Empresa** | Cargo | MM/AAAA - MM/AAAA' e depois "
        "bullets finais. Nunca use labels Cargo, Empresa, Periodo, Descricao ou "
        "Tecnologias dentro da experiencia.\n"
        "- Cada bullet comeca com verbo de acao e combina contexto, tecnologias "
        "concretas e impacto real. Experiencias nunca podem ficar cruas ou rasas.\n"
        "- Preserve a moldura factual: contribuicao de time continua colaborativa; "
        "autoria forte so quando estiver comprovada na propria experiencia. Nunca "
        "intensifique o verbo da fonte: Atuei, Contribui ou Participei nao podem "
        "virar Desenvolvi, Implementei, Liderei ou Construi.\n"
        "- Espelhe honestamente termos exatos da vaga quando eles existirem no "
        "historico. Nao use keyword stuffing.\n"
        "- Sem tabelas, colunas, icones, travessao Unicode ou parenteses decorativos.\n"
        "- Alvo de uma pagina: use no maximo 3 experiencias e 2 a 4 bullets densos "
        "por experiencia, priorizando os fatos mais relevantes sem esvaziar a "
        "substancia tecnica.\n"
        "- Nao crie conteudo para certificacoes ou idiomas quando o perfil e o "
        "contexto nao trouxerem esses fatos. Quando existirem, certificacoes usam "
        "bullets e idiomas ficam em uma linha separada por '|'.\n\n"
        "- Cada bullet comeca com um verbo de acao forte, encadeia o que foi feito, "
        "mostra o impacto real obtido e nomeia a tecnologia por extenso. Isso e "
        "estrutura da frase, nunca rotulo escrito no texto: jamais escreva as "
        "palavras 'resultado', 'ferramenta por extenso' ou qualquer nome de etapa "
        "dentro do bullet.\n"
        "- Nao afirme nenhuma tecnologia, empresa, metrica, autoria ou senioridade "
        "que nao exista no perfil-mestre ou no contexto factual deste request.\n\n"
        f"Erros a corrigir nesta tentativa:\n{reparos}\n\n"
        f"Perfil-mestre:\n{req.perfil_mestre.model_dump_json()}\n\n"
        f"Vaga: {req.vaga.titulo} @ {req.vaga.empresa}\n"
        f"Descricao da vaga:\n{req.vaga.descricao}\n\n"
        f"Palavras-chave a priorizar: {termos}\n\n"
        f"Contexto adicional:\n{chr(10).join(req.contexto)}\n\n"
        + (f"Markdown atual para reescrever sem perder fatos:\n{markdown_atual}\n" if markdown_atual else "")
    )


def _sem_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    ).lower()


def _contains_norm(texto: str, termo: str) -> bool:
    return _termo_presente(termo, texto)


def _fonte_factual(req: GenerateCvRequest) -> str:
    return "\n".join(
        [
            _texto_perfil(req.perfil_mestre),
            "\n".join(req.contexto),
        ]
    )


def _compactar_termo(texto: str) -> str:
    return re.sub(r"[^a-z0-9#+]+", "", normalize(texto))


def _termo_presente(termo: str, texto: str) -> bool:
    termo_norm = normalize(termo.strip())
    texto_norm = normalize(texto)
    if not termo_norm:
        return False
    padrao = rf"(?<![a-z0-9]){re.escape(termo_norm)}(?![a-z0-9])"
    if re.search(padrao, texto_norm):
        return True
    compacto = _compactar_termo(termo)
    return bool(len(compacto) > 2 and compacto in _compactar_termo(texto))


def _termo_bloqueado(termo: str, req: GenerateCvRequest) -> bool:
    return bool(termo.strip()) and not _termo_autorizado(termo, req)


def _termo_autorizado(termo: str, req: GenerateCvRequest) -> bool:
    return _termo_presente(termo, _fonte_factual(req))


def _limpar_termos_bloqueados(texto: str, req: GenerateCvRequest) -> str:
    return texto


def _titulo_vaga_seguro(titulo: str, req: GenerateCvRequest | None = None) -> str:
    limpo = re.sub(r"\s{2,}", " ", titulo).strip(" -/|,+")
    if req and any(
        _termo_presente(termo, limpo) and not _termo_autorizado(termo, req)
        for termo in TECH_CATALOG
    ):
        cargo = next((e.cargo for e in req.perfil_mestre.experiencias if e.cargo), "")
        if cargo:
            return cargo
        skill = next((s for s in req.perfil_mestre.skills if s), "")
        if skill:
            return f"Profissional de {skill}"
    return limpo or "Desenvolvedor Full-Stack"


def _url_contato(chave: str, valor: str) -> str | None:
    if chave.startswith("email"):
        return f"mailto:{valor}" if "@" in valor else None
    if chave.startswith(("linkedin", "github", "site")):
        return valor if "://" in valor else f"https://{valor}"
    return None


def _formatar_contato(chave: str, valor: str) -> str:
    url = _url_contato(chave, valor)
    return f"[{valor}]({url})" if url else valor


def _linha_contato(contato: dict[str, Any]) -> str:
    ordem = ["telefone", "email", "localizacao", "cidade", "site", "linkedin", "github"]
    usados: set[str] = set()
    partes: list[str] = []
    for chave in ordem:
        valor = contato.get(chave)
        if valor:
            partes.append(_formatar_contato(chave, str(valor)))
            usados.add(chave)
    partes.extend(
        _formatar_contato(k, str(v)) for k, v in contato.items() if k not in usados and v
    )
    return " | ".join(partes)


def _realizacoes(experiencia: ExperienciaPerfil) -> list[str]:
    if experiencia.realizacoes:
        return [item.strip().lstrip("- ").strip() for item in experiencia.realizacoes if item.strip()]
    linhas = [linha.strip() for linha in experiencia.descricao.splitlines() if linha.strip()]
    bullets = [m.group(1).strip() for linha in linhas if (m := BULLET_RE.match(linha))]
    if bullets:
        return bullets
    return [experiencia.descricao.strip()] if experiencia.descricao.strip() else []


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


def _periodo_mm_aaaa(periodo: str, idioma: str) -> str:
    atual = {"pt": "atual", "en": "present", "es": "actual"}[idioma]
    texto = periodo.strip()

    def substituir(match: re.Match[str]) -> str:
        mes = _sem_acentos(match.group(1)).rstrip(".")
        return f"{MESES.get(mes, match.group(1))}/{match.group(2)}"

    nomes = "|".join(sorted((re.escape(m) for m in MESES), key=len, reverse=True))
    texto = re.sub(rf"\b({nomes})\.?\s+(\d{{4}})\b", substituir, texto, flags=re.IGNORECASE)
    texto = re.sub(
        r"\s+(?:a|ate|até|to|hasta)\s+(?:atual|present|actual)\b",
        f" - {atual}", texto, flags=re.IGNORECASE,
    )
    return re.sub(r"\s*[–—]\s*", " - ", texto)


ATUAL_TERMOS = ("atual", "present", "actual")


def _parse_periodo(periodo: str) -> tuple[bool, tuple[int, int] | None, tuple[int, int] | None]:
    datas = [(int(ano), int(mes)) for mes, ano in re.findall(r"(\d{2})/(\d{4})", periodo)]
    inicio = datas[0] if datas else None
    atual = any(termo in _sem_acentos(periodo) for termo in ATUAL_TERMOS)
    if atual:
        return True, inicio, None
    fim = datas[1] if len(datas) > 1 else inicio
    return False, inicio, fim


def _chave_recencia(periodo: str) -> tuple[int, tuple[int, int], tuple[int, int]]:
    atual, inicio, fim = _parse_periodo(periodo)
    inicio = inicio or (0, 0)
    return (1 if atual else 0, inicio if atual else (fim or inicio), inicio)


def _nucleo_empresa(empresa: str) -> str:
    return re.split(r"\s*[·|,\-–—/]\s*", empresa.strip(), maxsplit=1)[0].strip()


def _experiencia_atual(experiencia: ExperienciaPerfil) -> bool:
    return _parse_periodo(experiencia.periodo)[0]


def _ordenar_experiencias(
    experiencias: list[ExperienciaPerfil], idioma: str
) -> list[ExperienciaPerfil]:
    return sorted(
        experiencias,
        key=lambda e: _chave_recencia(_periodo_mm_aaaa(e.periodo, idioma)),
        reverse=True,
    )


def _texto_perfil(perfil: PerfilMestre) -> str:
    partes = [
        perfil.nome, str(perfil.contato), perfil.resumo,
        " ".join(_texto_experiencia(e) for e in perfil.experiencias),
        " ".join(perfil.formacao), " ".join(perfil.certificacoes),
        " ".join(perfil.idiomas), " ".join(perfil.skills),
    ]
    return "\n".join(p for p in partes if p)


def _idioma(req: GenerateCvRequest) -> str:
    texto = normalize(f"{req.vaga.titulo} {req.vaga.descricao}")
    en = ["requirements", "responsibilities", "experience", "english", "skills", "we are", "hiring"]
    es = [
        "espanol", "desarrollador", "desarrolladora", "conocimientos",
        "habilidades", "formacion", "trabajo remoto", "postulate",
    ]
    if sum(t in texto for t in en) >= 2:
        return "en"
    if "español" in req.vaga.descricao.lower() or sum(t in texto for t in es) >= 2:
        return "es"
    return "pt"


def _cabecalhos(idioma: str) -> dict[str, str]:
    if idioma == "en":
        return {
            "resumo": "PROFESSIONAL SUMMARY", "competencias": "SKILLS",
            "experiencia": "PROFESSIONAL EXPERIENCE", "formacao": "EDUCATION",
            "certificacoes": "CERTIFICATIONS", "idiomas": "LANGUAGES",
        }
    if idioma == "es":
        return {
            "resumo": "RESUMEN PROFESIONAL", "competencias": "COMPETENCIAS",
            "experiencia": "EXPERIENCIA PROFESIONAL", "formacao": "FORMACIÓN ACADÉMICA",
            "certificacoes": "CERTIFICACIONES", "idiomas": "IDIOMAS",
        }
    return {
        "resumo": "RESUMO PROFISSIONAL", "competencias": "COMPETÊNCIAS",
        "experiencia": "EXPERIÊNCIA PROFISSIONAL", "formacao": "FORMAÇÃO ACADÊMICA",
        "certificacoes": "CERTIFICAÇÕES", "idiomas": "IDIOMAS",
    }


def _texto_experiencia(experiencia: ExperienciaPerfil) -> str:
    partes = [experiencia.cargo, experiencia.empresa, experiencia.periodo, experiencia.local]
    partes.extend(_realizacoes(experiencia))
    partes.extend(experiencia.tecnologias)
    return "\n".join(parte for parte in partes if parte)


def _competencias_relevantes(req: GenerateCvRequest) -> list[str]:
    perfil_texto = normalize(_texto_perfil(req.perfil_mestre))
    relevantes = [
        k.termo.strip() for k in req.keywords
        if k.termo.strip() and normalize(k.termo) in perfil_texto and _termo_autorizado(k.termo, req)
    ]
    for skill in req.perfil_mestre.skills:
        skill_limpa = _limpar_termos_bloqueados(skill, req)
        if skill_limpa and skill_limpa not in relevantes and _termo_autorizado(skill_limpa, req):
            relevantes.append(skill_limpa)
    return relevantes[:24]


def _resumo_tailored(req: GenerateCvRequest, competencias: list[str], idioma: str) -> str:
    titulo = _titulo_vaga_seguro(req.vaga.titulo, req)
    top = ", ".join(competencias[:8]) or "desenvolvimento full-stack, APIs RESTful e sistemas corporativos"
    if idioma == "en":
        return (
            f"Full-stack developer aligned with {titulo}, with factual experience in "
            f"{top}, production systems, RESTful APIs and collaborative delivery."
        )
    if idioma == "es":
        return (
            f"Desarrollador full-stack alineado con {titulo}, con experiencia factual "
            f"en {top}, sistemas en produccion, APIs RESTful y trabajo colaborativo."
        )
    return (
        f"Desenvolvedor full-stack alinhado a {titulo}, com experiencia factual em "
        f"{top}, sistemas em producao, APIs RESTful e entrega colaborativa em times "
        "de produto."
    )


def _categorizar_competencias(competencias: list[str], idioma: str) -> list[tuple[str, list[str]]]:
    rotulos = {
        "pt": ["Linguagens", "Backend e APIs", "Frontend", "Dados", "Cloud e DevOps", "IA e Automacao", "Arquitetura e Qualidade", "Outras"],
        "en": ["Languages", "Backend and APIs", "Frontend", "Data", "Cloud and DevOps", "AI and Automation", "Architecture and Quality", "Other"],
        "es": ["Lenguajes", "Backend y APIs", "Frontend", "Datos", "Cloud y DevOps", "IA y Automatizacion", "Arquitectura y Calidad", "Otras"],
    }[idioma]
    grupos: list[list[str]] = [[] for _ in rotulos]
    regras = [
        (0, ("python", "typescript", "javascript", "java", "c#", "go", "ruby", "php")),
        (1, ("api", "fastapi", "nestjs", "node", "spring", ".net", "django", "flask", "jwt", "microserv")),
        (2, ("react", "next", "angular", "vue", "html", "css", "tailwind", "vite", "frontend")),
        (3, ("postgres", "mysql", "mongo", "redis", "database", "banco", "sql", "prisma")),
        (4, ("aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "github actions", "cloud")),
        (5, ("llm", "rag", "agente", "agent", "ocr", "yolo", "opencv", "machine learning", "ia", "ai")),
        (6, ("clean architecture", "solid", "design pattern", "teste", "test", "scrum", "kanban", "code review")),
    ]
    for competencia in competencias:
        n = normalize(competencia)
        indice = next((i for i, termos in regras if any(t in n for t in termos)), len(rotulos) - 1)
        grupos[indice].append(competencia)
    return [(rotulos[i], grupo) for i, grupo in enumerate(grupos) if grupo]


def _selecionar_realizacoes(experiencia: ExperienciaPerfil, req: GenerateCvRequest, limite: int) -> list[str]:
    itens = _realizacoes(experiencia)
    if len(itens) <= limite:
        return itens
    termos = [normalize(k.termo) for k in req.keywords if k.termo.strip()]
    pontuados = []
    for indice, item in enumerate(itens):
        texto = normalize(item)
        pontos = sum(1 for termo in termos if termo in texto)
        pontuados.append((pontos, -indice, indice))
    escolhidos = sorted(i for _, _, i in sorted(pontuados, reverse=True)[:limite])
    return [itens[i] for i in escolhidos]


def _deterministic_request(req: GenerateCvRequest) -> str:
    perfil = req.perfil_mestre
    idioma = _idioma(req)
    h = _cabecalhos(idioma)
    titulo = _titulo_vaga_seguro(
        req.vaga.titulo or (perfil.skills[0] if perfil.skills else "Profissional"),
        req,
    )
    linhas = [f"# {perfil.nome}".strip(), f"**{titulo}**"]
    contato = _linha_contato(perfil.contato)
    if contato:
        linhas += ["", contato]

    competencias = _competencias_relevantes(req)
    linhas += ["", f"## {h['resumo']}", _resumo_tailored(req, competencias, idioma)]
    linhas += ["", f"## {h['competencias']}"]
    for categoria, valores in _categorizar_competencias(competencias, idioma):
        linhas.append(f"- {categoria}: {', '.join(valores)}")

    linhas += ["", f"## {h['experiencia']}"]
    for indice, experiencia in enumerate(_ordenar_experiencias(perfil.experiencias, idioma)[:3]):
        empresa = experiencia.empresa or "Empresa"
        cargo = experiencia.cargo or "Cargo"
        periodo = _periodo_mm_aaaa(experiencia.periodo, idioma) or "periodo nao informado"
        linhas += ["", f"**{empresa}** | {cargo} | {periodo}"]
        limite = 3 if indice < 2 else 2
        for realizacao in _selecionar_realizacoes(experiencia, req, limite):
            limpa = _limpar_termos_bloqueados(realizacao, req)
            if limpa:
                linhas.append(f"- {limpa}")

    linhas += ["", f"## {h['formacao']}", *perfil.formacao]
    linhas += ["", f"## {h['certificacoes']}", *[f"- {item}" for item in perfil.certificacoes]]
    linhas += ["", f"## {h['idiomas']}"]
    if perfil.idiomas:
        linhas.append(" | ".join(perfil.idiomas))
    return "\n".join(linhas).strip()


def _normalizar_cabecalhos(texto: str, req: GenerateCvRequest) -> str:
    h = _cabecalhos(_idioma(req))
    sinonimos = {
        "resumo profissional": h["resumo"], "perfil profissional": h["resumo"], "resumen profesional": h["resumo"], "professional summary": h["resumo"],
        "competencias": h["competencias"], "habilidades": h["competencias"], "skills": h["competencias"],
        "experiencia profissional": h["experiencia"], "experiencia profesional": h["experiencia"], "experiencias": h["experiencia"], "professional experience": h["experiencia"],
        "formacao academica": h["formacao"], "formacion academica": h["formacao"], "educacao": h["formacao"], "education": h["formacao"],
        "certificacoes": h["certificacoes"], "certificaciones": h["certificacoes"], "certifications": h["certificacoes"],
        "idiomas": h["idiomas"], "languages": h["idiomas"],
    }
    linhas = []
    for linha in texto.splitlines():
        if linha.startswith("## "):
            chave = _sem_acentos(linha[3:].strip())
            linha = f"## {sinonimos.get(chave, linha[3:].strip())}"
        linhas.append(linha.rstrip())
    return "\n".join(linhas)


def _normalizar_cabecalho_experiencia(texto: str, req: GenerateCvRequest) -> str:
    idioma = _idioma(req)
    linhas = []
    for linha in texto.splitlines():
        bruta = linha.strip()
        m = re.match(r"^#{1,6}\s+(.+)$", bruta)
        conteudo = m.group(1).strip() if m else bruta
        partes = [p.strip() for p in conteudo.split("|")]
        if m and len(partes) >= 3:
            primeira = partes[0]
            if not (primeira.startswith("**") and primeira.endswith("**")):
                primeira = f"**{primeira.strip('*').strip()}**"
            periodo = _periodo_mm_aaaa(partes[2], idioma)
            linhas.append(" | ".join([primeira, partes[1], periodo]))
        else:
            linhas.append(linha)
    return "\n".join(linhas)


def _limpar_markdown(markdown: str, req: GenerateCvRequest) -> str:
    texto = markdown.translate(PONTUACAO_ASCII)
    texto = "\n".join(re.sub(r"[ \t]+", " ", linha).rstrip() for linha in texto.splitlines())
    texto = _normalizar_cabecalho_experiencia(texto, req)
    linhas = _normalizar_cabecalhos(texto, req).strip().splitlines()
    if linhas and linhas[0].startswith("# ") and " | " in linhas[0]:
        nome, titulo = linhas[0][2:].split(" | ", 1)
        linhas[0:1] = [f"# {nome.strip()}", f"**{titulo.strip()}**"]
    uteis = [i for i, linha in enumerate(linhas) if linha.strip()]
    if uteis and not any(linhas[i].startswith("**") and linhas[i].endswith("**") for i in uteis[1:3]):
        linhas.insert(uteis[0] + 1, f"**{_titulo_vaga_seguro(req.vaga.titulo or 'Profissional', req)}**")
    uteis = [i for i, linha in enumerate(linhas) if linha.strip()]
    if len(uteis) >= 2 and req.vaga.titulo:
        linhas[uteis[1]] = f"**{_titulo_vaga_seguro(req.vaga.titulo, req)}**"
    contato = _linha_contato(req.perfil_mestre.contato)
    uteis = [i for i, linha in enumerate(linhas) if linha.strip()]
    if len(uteis) >= 3 and contato and not linhas[uteis[2]].startswith("#"):
        linhas[uteis[2]] = contato
    return "\n".join(linhas).strip()


def _secao(markdown: str, titulo: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(titulo)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        markdown, re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _erros_contrato(markdown: str, req: GenerateCvRequest) -> list[str]:
    erros: list[str] = []
    h = _cabecalhos(_idioma(req))
    linhas_uteis = [linha.strip() for linha in markdown.splitlines() if linha.strip()]
    if not linhas_uteis or not re.fullmatch(r"#\s+[^|]+", linhas_uteis[0]):
        erros.append("cabecalho de nome invalido")
    if len(linhas_uteis) < 2 or not re.fullmatch(r"\*\*.+\*\*", linhas_uteis[1]):
        erros.append("titulo profissional ausente")
    if len(linhas_uteis) < 3 or linhas_uteis[2].startswith("#"):
        erros.append("linha de contato ausente")
    posicoes = []
    for secao in h.values():
        marcador = f"## {secao}"
        pos = markdown.upper().find(marcador.upper())
        if pos < 0:
            erros.append(f"secao ausente: {secao}")
        posicoes.append(pos)
    if all(pos >= 0 for pos in posicoes) and posicoes != sorted(posicoes):
        erros.append("ordem de secoes invalida")
    competencias = _secao(markdown, h["competencias"])
    if competencias and not any(SKILL_RE.match(linha.strip()) for linha in competencias.splitlines()):
        erros.append("competencias sem categorias")
    certificacoes = _secao(markdown, h["certificacoes"])
    if req.perfil_mestre.certificacoes and not any(
        BULLET_RE.match(linha.strip()) for linha in certificacoes.splitlines()
    ):
        erros.append("certificacoes sem bullets")
    experiencias = _secao(markdown, h["experiencia"])
    if req.perfil_mestre.experiencias:
        if not any(JOB_HEADER_RE.match(linha.strip()) for linha in experiencias.splitlines()):
            erros.append("cabecalho de experiencia invalido")
        if not any(BULLET_RE.match(linha.strip()) for linha in experiencias.splitlines()):
            erros.append("experiencias sem bullets")
    if re.search(
        r"^(?:Cargo|Empresa|Per[ií]odo|Descri[cç][aã]o|Tecnologias)\s*:",
        experiencias, re.MULTILINE | re.IGNORECASE,
    ):
        erros.append("labels crus na experiencia")
    return erros


def _erros_factualidade(markdown: str, req: GenerateCvRequest) -> list[str]:
    alucinadas = sorted(
        {
            termo
            for termo in TECH_CATALOG
            if _termo_presente(termo, markdown) and not _termo_autorizado(termo, req)
        },
        key=str.lower,
    )
    if not alucinadas:
        return []
    return [
        "tecnologia sem fonte factual no perfil/contexto do usuario: "
        + ", ".join(alucinadas)
    ]


def _erros_formula(markdown: str) -> list[str]:
    normalizado = normalize(markdown)
    vazamentos = [termo for termo in FORMULA_LEAK_TERMS if termo in normalizado]
    if FORMULA_LABEL_RE.search(markdown):
        vazamentos.append("resultado:")
    if not vazamentos:
        return []
    return [
        "vocabulario interno da formula de bullet vazou na saida: "
        + ", ".join(vazamentos)
    ]


def _erros_coerencia(markdown: str, req: GenerateCvRequest) -> list[str]:
    h = _cabecalhos(_idioma(req))
    resumo = _secao(markdown, h["resumo"])
    if not resumo:
        return []
    alvo = set(content_tokens(req.vaga.titulo)) | {
        token for k in req.keywords for token in content_tokens(k.termo)
    }
    if not alvo or alvo & set(content_tokens(resumo)):
        return []
    return ["resumo profissional sem vocabulario em comum com a vaga"]


def _erros_completude(markdown: str, req: GenerateCvRequest) -> list[str]:
    experiencias = req.perfil_mestre.experiencias
    if not experiencias:
        return []
    h = _cabecalhos(_idioma(req))
    secao = _secao(markdown, h["experiencia"])
    if len(experiencias) <= 3:
        faltando = [
            e.empresa
            for e in experiencias
            if e.empresa.strip() and not _termo_presente(_nucleo_empresa(e.empresa), secao)
        ]
        if faltando:
            return ["experiencia do perfil-mestre ausente na saida: " + ", ".join(faltando)]
        return []
    faltando = [
        e.empresa
        for e in experiencias
        if _experiencia_atual(e)
        and e.empresa.strip()
        and not _termo_presente(_nucleo_empresa(e.empresa), secao)
    ]
    if faltando:
        return ["experiencia mais recente do perfil-mestre omitida: " + ", ".join(faltando)]
    return []


def _erros_ordem(markdown: str, req: GenerateCvRequest) -> list[str]:
    h = _cabecalhos(_idioma(req))
    secao = _secao(markdown, h["experiencia"])
    chaves = [
        _chave_recencia(m.group(3))
        for linha in secao.splitlines()
        if (m := JOB_HEADER_RE.match(linha.strip()))
    ]
    if len(chaves) < 2 or chaves == sorted(chaves, reverse=True):
        return []
    return [
        "experiencias fora de ordem cronologica reversa; liste a mais recente "
        "primeiro e mantenha a experiencia atual antes das ja encerradas"
    ]


def _erros_saida(markdown: str, req: GenerateCvRequest) -> list[str]:
    return [
        *_erros_contrato(markdown, req),
        *_erros_factualidade(markdown, req),
        *_erros_formula(markdown),
        *_erros_coerencia(markdown, req),
        *_erros_completude(markdown, req),
        *_erros_ordem(markdown, req),
    ]


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


def _lacunas_autorizadas(analise: AtsAnalysis, req: GenerateCvRequest) -> list[str]:
    return [
        termo
        for termo in analise.keywords_criticas_ausentes
        if _termo_autorizado(termo, req)
    ]


def _gerar_llm(
    req: GenerateCvRequest,
    analise: AtsAnalysis,
    lacunas: list[str],
    markdown_atual: str | None = None,
    erros: list[str] | None = None,
) -> str:
    res = complete_model(
        _system_prompt(),
        _user(req, analise, lacunas, markdown_atual=markdown_atual, erros=erros),
        GenerateCvResponse,
    )
    return _limpar_markdown(res.markdown, req) if res.markdown.strip() else ""


def generate_cv(req: GenerateCvRequest) -> str:
    return generate_cv_pipeline(req).markdown


def analisar_ats(req: GenerateCvRequest) -> AtsAnalysis:
    if not req.keywords:
        raise KeywordsUnavailable(
            "extracao de keywords pendente; tente novamente antes de analisar"
        )
    return _analise(_deterministic_request(req), req)


def generate_cv_pipeline(req: GenerateCvRequest) -> GeneratePipelineResponse:
    if not req.keywords:
        raise KeywordsUnavailable(
            "extracao de keywords pendente; tente novamente antes de gerar o curriculo"
        )
    base = _deterministic_request(req)
    inicial = _analise(base, req)
    degradacao = None
    markdown = ""
    erros = []
    try:
        lacunas = _lacunas_autorizadas(inicial, req)
        for tentativa in range(2):
            markdown = _gerar_llm(req, inicial, lacunas, erros=erros)
            erros = _erros_saida(markdown, req) if markdown else ["resposta vazia"]
            if erros:
                logging.getLogger(__name__).warning(
                    "reescrita rejeitada tentativa=%s erros=%s", tentativa + 1, erros
                )
            if not erros:
                break
        if erros:
            markdown = base
            degradacao = "Geracao degradada para fallback factual: " + "; ".join(erros)
        else:
            parcial = _analise(markdown, req)
            reforcos = _lacunas_autorizadas(parcial, req)
            if parcial.score < 75 and reforcos:
                reparos = [
                    "score final abaixo de 75; reescreva para fechar lacunas "
                    f"factuais sem stuffing: {', '.join(reforcos)}"
                ]
                try:
                    candidato = _gerar_llm(
                        req,
                        parcial,
                        reforcos,
                        markdown_atual=markdown,
                        erros=reparos,
                    )
                    erros_reforco = _erros_saida(candidato, req) if candidato else ["resposta vazia"]
                    if erros_reforco:
                        degradacao = (
                            "Passe dirigido nao aplicado; mantida versao factual valida: "
                            + "; ".join(erros_reforco)
                        )
                    else:
                        markdown = candidato
                except LLMUnavailable as exc:
                    degradacao = (
                        "Passe dirigido indisponivel; mantida versao factual valida: "
                        f"{exc}"
                    )
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


ERRO_EXCESSO_PAGINA = (
    "o curriculo excedeu uma pagina mesmo apos compactar o layout; reduza para no "
    "maximo 2 a 3 bullets por experiencia e enxugue os menos densos, sem esvaziar a "
    "substancia tecnica. So remova uma experiencia inteira se o perfil-mestre tiver "
    "mais de 3 experiencias, e nesse caso remova a menos aderente a vaga, nunca a "
    "mais recente. Preserve todas as experiencias do perfil-mestre quando forem 3 ou "
    "menos."
)


def reduzir_curriculo(
    req: GenerateCvRequest, markdown_atual: str
) -> GeneratePipelineResponse:
    if not req.keywords:
        raise KeywordsUnavailable(
            "extracao de keywords pendente; tente novamente antes de reduzir o curriculo"
        )
    inicial = _analise(_deterministic_request(req), req)
    markdown = markdown_atual
    degradacao = None
    try:
        candidato = _gerar_llm(
            req,
            inicial,
            _lacunas_autorizadas(inicial, req),
            markdown_atual=markdown_atual,
            erros=[ERRO_EXCESSO_PAGINA],
        )
        erros = _erros_saida(candidato, req) if candidato else ["resposta vazia"]
        if erros:
            degradacao = (
                "Corte de conteudo nao aplicado; mantida versao anterior: "
                + "; ".join(erros)
            )
        else:
            markdown = candidato
    except LLMUnavailable as exc:
        degradacao = f"Corte de conteudo indisponivel: {exc}"
    final = _analise(markdown, req)
    return GeneratePipelineResponse(
        markdown=markdown,
        analise_inicial=inicial,
        analise_final=final,
        degradacao=degradacao,
    )
