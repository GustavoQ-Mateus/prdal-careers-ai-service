from .schemas import ClassifyResponse
from .text import normalize, tokens

CATEGORIA_SINAIS: dict[str, list[tuple[str, int]]] = {
    "ia": [
        ("machine learning", 3), ("inteligencia artificial", 3), ("ai engineer", 3),
        ("deep learning", 3), ("computer vision", 2), ("visao computacional", 2),
        ("llm", 2), ("nlp", 2), ("ml", 2), ("pytorch", 2), ("tensorflow", 2),
        ("genai", 2), ("rag", 1),
    ],
    "dados": [
        ("data engineer", 3), ("data scientist", 3), ("engenheiro de dados", 3),
        ("cientista de dados", 3), ("power bi", 2), ("etl", 2), ("spark", 2),
        ("hadoop", 2), ("airflow", 2), ("analytics", 2), ("warehouse", 2),
        ("dbt", 2), ("dados", 2), ("data", 1), ("pandas", 1),
    ],
    "mobile": [
        ("react native", 3), ("flutter", 2), ("android", 2), ("ios", 2),
        ("kotlin", 2), ("swift", 2), ("mobile", 2),
    ],
    "devops": [
        ("devops", 3), ("sre", 3), ("kubernetes", 2), ("terraform", 2),
        ("ci/cd", 2), ("jenkins", 2), ("ansible", 2), ("infraestrutura", 2),
        ("infra", 1), ("cloud", 1),
    ],
    "qa": [
        ("quality assurance", 3), ("qa", 3), ("cypress", 2), ("selenium", 2),
        ("playwright", 2), ("tester", 2), ("testes", 1), ("teste", 1),
        ("qualidade", 1),
    ],
    "design": [
        ("ux designer", 3), ("ui designer", 3), ("product designer", 3),
        ("figma", 3), ("ui/ux", 2), ("designer", 2),
    ],
    "produto": [
        ("product manager", 3), ("product owner", 3), ("gerente de produto", 3),
        ("gestao de produto", 2), ("pm", 1), ("po", 1),
    ],
    "backend": [
        ("backend", 2), ("back-end", 2), ("back end", 2), ("node.js", 2),
        ("nodejs", 2), ("node", 2), ("nestjs", 2), ("django", 2), ("flask", 2),
        ("spring", 2), (".net", 2), ("c#", 2), ("java", 2), ("php", 2),
        ("laravel", 2), ("rails", 2), ("golang", 2), ("microservicos", 2),
        ("microservices", 2), ("express", 1), ("api", 1), ("kafka", 1),
        ("rabbitmq", 1),
    ],
    "frontend": [
        ("front-end", 2), ("front end", 2), ("frontend", 2), ("react", 2),
        ("angular", 2), ("vue", 2), ("svelte", 2), ("tailwind", 2),
        ("next.js", 2), ("nextjs", 2), ("nuxt", 2), ("css", 1), ("html", 1),
    ],
}

FULLSTACK_TERMS = ("fullstack", "full stack", "full-stack")

ORDEM_DESEMPATE = [
    "ia", "dados", "mobile", "devops", "qa", "design", "produto",
    "backend", "frontend",
]

NIVEL_SINAIS: list[tuple[str, tuple[str, ...]]] = [
    ("estagio", ("estagio", "estagiario", "intern", "internship")),
    ("senior", ("senior", "especialista", "tech lead", "staff", "principal", "sr")),
    ("pleno", ("pleno", "mid-level", "mid level", "pl")),
    ("junior", ("junior", "trainee", "entry level", "entry-level", "jr")),
]


def _casa(termo: str, texto_norm: str, tokset: set[str]) -> bool:
    if termo.isalnum():
        return termo in tokset
    return termo in texto_norm


def _pontuar(texto_norm: str, tokset: set[str]) -> dict[str, int]:
    pontos = {}
    for categoria, sinais in CATEGORIA_SINAIS.items():
        pontos[categoria] = sum(
            peso for termo, peso in sinais if _casa(termo, texto_norm, tokset)
        )
    return pontos


def _categoria(texto_norm: str, tokset: set[str]) -> str:
    if any(t in texto_norm for t in FULLSTACK_TERMS):
        return "fullstack"

    pontos = _pontuar(texto_norm, tokset)
    if pontos["frontend"] >= 2 and pontos["backend"] >= 2:
        return "fullstack"

    melhor = max(ORDEM_DESEMPATE, key=lambda c: pontos[c])
    return melhor if pontos[melhor] > 0 else "outro"


def _nivel(texto_norm: str, tokset: set[str]) -> str:
    for nivel, termos in NIVEL_SINAIS:
        if any(_casa(t, texto_norm, tokset) for t in termos):
            return nivel
    return "indefinido"


def classificar(titulo: str, descricao: str) -> ClassifyResponse:
    bruto = f"{titulo} {descricao}"
    texto_norm = normalize(bruto)
    tokset = set(tokens(bruto))
    return ClassifyResponse(
        categoria=_categoria(texto_norm, tokset),
        nivel=_nivel(texto_norm, tokset),
    )
