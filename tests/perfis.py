import json

from app.schemas import GenerateCvRequest

PERFIL_DEV = {
    "nome": "Pessoa Exemplo",
    "emails": [{"valor": "pessoa@exemplo.dev", "principal": True}],
    "telefones": [{"ddi": "+55", "numero": "11 90000-0000", "principal": True}],
    "links": [{"tipo": "linkedin", "url": "linkedin.com/in/pessoa-exemplo"}],
    "endereco": {"pais": "Brasil", "estado": "SP", "cidade": "Campinas"},
    "resumo": "Desenvolvedora back-end com APIs REST em Python e Java.",
    "experiencias": [
        {
            "id": "rota",
            "empresa": "Rota Inspecoes",
            "cargo": "Desenvolvedora Back-end",
            "dataInicioMes": 6, "dataInicioAno": 2025, "atual": True,
            "descricao": (
                "- Atuei no back-end de plataforma web em producao com Python (FastAPI) e PostgreSQL.\n"
                "- Implementei autenticacao JWT multi-tenant e filas assincronas.\n"
                "- Reduzi o tempo de resposta das consultas em 40% com indices no PostgreSQL."
            ),
        },
        {
            "id": "erp",
            "empresa": "Sistemas Gestao",
            "cargo": "Estagiaria de Desenvolvimento",
            "dataInicioMes": 1, "dataInicioAno": 2024, "dataFimMes": 4, "dataFimAno": 2025,
            "descricao": "- Atuei em modulos ERP com Java (Spring Boot) sobre MySQL.",
        },
    ],
    "formacao": [{"instituicao": "Universidade Exemplo", "curso": "ADS", "inicioMes": 2, "inicioAno": 2023, "fimMes": 6, "fimAno": 2026}],
    "certificacoes": [{"titulo": "Certificacao Exemplo", "descricao": "Escola Exemplo, 2025"}],
    "idiomas": ["Portugues, nativo", "Ingles, intermediario"],
    "skills": ["Java", "Spring Boot", "Python", "FastAPI", "PostgreSQL", "MySQL"],
}

VAGA_DEV = {
    "titulo": "Desenvolvedora Back-End Java",
    "empresa": "Empresa Vaga",
    "descricao": "Buscamos Java, Spring Boot, APIs REST, MySQL e Kubernetes para sistemas corporativos.",
}

KEYWORDS_DEV = [
    {"termo": "Java", "peso": 1},
    {"termo": "Spring Boot", "peso": 0.9},
    {"termo": "MySQL", "peso": 0.8},
    {"termo": "APIs REST", "peso": 0.7},
    {"termo": "Kubernetes", "peso": 0.6},
]

PERFIL_DADOS = {
    "nome": "Analista Exemplo",
    "emails": [{"valor": "analista@exemplo.dev", "principal": True}],
    "resumo": "Analista de dados com foco em indicadores comerciais, SQL e Power BI.",
    "experiencias": [
        {
            "id": "varejo",
            "empresa": "Varejo Exemplo",
            "cargo": "Analista de Dados",
            "dataInicioMes": 3, "dataInicioAno": 2023, "atual": True,
            "descricao": (
                "- Construi paineis de vendas em Power BI para a diretoria comercial.\n"
                "- Escrevi consultas SQL para consolidar dados de 120 lojas.\n"
                "- Automatizei relatorios semanais em Excel."
            ),
        },
        {
            "id": "banco",
            "empresa": "Banco Exemplo",
            "cargo": "Assistente de Dados",
            "dataInicioMes": 1, "dataInicioAno": 2021, "dataFimMes": 2, "dataFimAno": 2023,
            "descricao": "- Mantive planilhas de conciliacao em Excel.\n- Apoiei a area de risco com relatorios mensais.",
        },
    ],
    "formacao": [{"grau": "Bacharelado", "instituicao": "Universidade Exemplo", "curso": "Estatistica", "inicioAno": 2017, "fimAno": 2021, "status": "concluido"}],
    "idiomas": ["Portugues, nativo"],
    "skills": ["SQL", "Power BI", "Excel", "Estatistica"],
}

VAGA_DADOS = {
    "titulo": "Analista de BI",
    "empresa": "Empresa Dados",
    "descricao": "Analista de BI com SQL, Power BI e Excel para indicadores comerciais.",
}

KEYWORDS_DADOS = [
    {"termo": "SQL", "peso": 1},
    {"termo": "Power BI", "peso": 0.9},
    {"termo": "Excel", "peso": 0.6},
]

VOCABULARIO_DEV = (
    "desenvolved", "full-stack", "full stack", "backend", "back-end", "frontend", "apis",
    "linguagens", "devops", "cloud", "ia e automacao", "software",
)


def req_dev(contexto=None, **vaga) -> GenerateCvRequest:
    return GenerateCvRequest.model_validate(
        {
            "perfilMestre": PERFIL_DEV,
            "vaga": {**VAGA_DEV, **vaga},
            "keywords": KEYWORDS_DEV,
            "contexto": contexto or [],
        }
    )


def req_dados() -> GenerateCvRequest:
    return GenerateCvRequest.model_validate(
        {"perfilMestre": PERFIL_DADOS, "vaga": VAGA_DADOS, "keywords": KEYWORDS_DADOS}
    )


def frase(texto, *fontes):
    return {"texto": texto, "fontes": list(fontes)}


def reescrita(titulo, resumo, experiencias, competencias=None):
    return {
        "titulo": titulo,
        "resumo": resumo,
        "experiencias": [{"experienciaId": i, "bullets": b} for i, b in experiencias],
        "competencias": competencias or [],
        "reparos": [],
    }


def reparo(*frases):
    return {
        "titulo": {"texto": "", "fontes": []},
        "resumo": [],
        "experiencias": [],
        "competencias": [],
        "reparos": [{"chave": c, "texto": t, "fontes": list(f)} for c, t, f in frases],
    }


def json_texto(dados) -> str:
    return json.dumps(dados, ensure_ascii=False)
