RESUMO_MAX_FRASES = 3
EXPERIENCIAS_RECENTES = 2
BULLETS_RECENTES = 4
BULLETS_DEMAIS = 3
BULLETS_TOTAL = 12
PISO_BULLETS = 2
COMPETENCIAS_MAX_CATEGORIAS = 5
COMPETENCIAS_MAX_TERMOS = 20
EXPERIENCIAS_SEM_CORTE = 3

DESCRICAO_TITULO = (
    "Titulo profissional proposto para a vaga, curto e sem nome de empresa, "
    "sustentado pelas fontes citadas"
)
DESCRICAO_FONTES = (
    "Ids das fontes factuais que sustentam a frase inteira; nunca cite fonte de apoio"
)
DESCRICAO_RESUMO = (
    f"Frases do resumo profissional, no maximo {RESUMO_MAX_FRASES}, da mais para a "
    "menos relevante para a vaga; cada frase cita as fontes que a sustentam"
)
DESCRICAO_EXPERIENCIAS = (
    "Uma entrada por experiencia do perfil que tenha algo relevante para a vaga, "
    "usando o experienciaId informado"
)
DESCRICAO_BULLETS = (
    f"Bullets da experiencia, da mais para a menos relevante para a vaga: no maximo "
    f"{BULLETS_RECENTES} nas {EXPERIENCIAS_RECENTES} experiencias mais recentes e no "
    f"maximo {BULLETS_DEMAIS} nas demais, com no maximo {BULLETS_TOTAL} bullets no "
    "curriculo inteiro para caber em uma pagina; cada bullet cita a propria "
    "experiencia ou uma nota factual"
)
DESCRICAO_COMPETENCIAS = (
    f"Ate {COMPETENCIAS_MAX_CATEGORIAS} categorias de competencias, com no maximo "
    f"{COMPETENCIAS_MAX_TERMOS} termos no total, dos mais para os menos relevantes "
    "para a vaga; cada termo cita a fonte onde aparece escrito"
)

DESCRICAO_REPAROS = (
    "Vazio na geracao. No pedido de reparo, uma entrada por frase rejeitada, e os "
    "demais campos ficam vazios"
)


def bullets_maximos(posicao_por_recencia: int) -> int:
    return BULLETS_RECENTES if posicao_por_recencia < EXPERIENCIAS_RECENTES else BULLETS_DEMAIS


def texto_orcamento() -> str:
    return (
        f"- Resumo: no maximo {RESUMO_MAX_FRASES} frases.\n"
        f"- Experiencias: no maximo {BULLETS_RECENTES} bullets em cada uma das "
        f"{EXPERIENCIAS_RECENTES} experiencias mais recentes e no maximo "
        f"{BULLETS_DEMAIS} nas demais.\n"
        f"- Total: no maximo {BULLETS_TOTAL} bullets no curriculo, para caber em uma pagina.\n"
        f"- Competencias: ate {COMPETENCIAS_MAX_CATEGORIAS} categorias e "
        f"{COMPETENCIAS_MAX_TERMOS} termos.\n"
        "- O que passar do orcamento e cortado pelo sistema a partir do fim de cada "
        "lista, por isso ordene sempre do mais para o menos relevante."
    )
