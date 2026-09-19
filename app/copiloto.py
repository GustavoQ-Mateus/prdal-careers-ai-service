import json
import re
import unicodedata

from .llm import LLMUnavailable, complete_model
from .schemas import (
    PerfilMestre,
    RedigirFormularioRequest,
    RedigirFormularioResponse,
    RedigirMensagemRequest,
    RedigirMensagemResponse,
    TurnRequest,
    TurnResponse,
    Vaga,
)

SYSTEM_TURNO = (
    "Voce e o copiloto de candidatura, um agente que conduz o candidato pela "
    "preparacao de vagas. Raciocina e escolhe o proximo passo, mas quem executa "
    "as tools e a api; voce so declara a intencao. Use apenas as tools listadas. "
    "Aja uma tool por vez. Peca leitura antes de escrita. O contexto de oportunidade "
    "em foco e a maquina de estados da conversa sao autoridade: depois de registrar "
    "uma oportunidade, use o id retornado para os proximos passos e nunca registre "
    "a mesma vaga novamente. Uma tool que ja aparece como concluida no historico nao "
    "deve ser chamada de novo. Nunca invente numero de "
    "score; o score vem sempre da tool. Nunca envie nada externo por conta propria; "
    "para mensagem a recrutador ou resposta de formulario, use as tools de redacao "
    "que entregam texto ao candidato revisar. Sequencia interna obrigatoria de "
    "redacao: quando o candidato pedir uma mensagem ao recrutador, chame "
    "redigir_mensagem_recrutador usando o perfil-mestre e a vaga em foco; quando "
    "pedir respostas de formulario, pergunte ou use os nomes dos campos e chame "
    "redigir_respostas_formulario. Entregue o texto para revisao e nunca trate "
    "redacao como envio ou candidatura concluida. "
    "curriculo: primeiro, registrar ou revisar a vaga; depois, iniciar "
    "gerar_curriculo uma unica vez; por fim, consultar status_geracao usando o "
    "jobId retornado e, somente quando o status for CONCLUIDA, chamar "
    "buscar_curriculo com o curriculoId para ler o curriculo, score e breakdown "
    "final. Nao avance para mensagem, formulario, candidatura ou proximo passo "
    "externo antes de concluir essa consulta final. Quando buscar_curriculo trouxer "
    "analiseInicial e analiseFinal apos uma geracao CONCLUIDA, responda ao candidato "
    "em duas mensagens de texto sequenciais, nunca com um resumo de uma linha. A "
    "primeira e 'Etapa 1 — Analise ATS': informe score, keywordsEncontradas, "
    "keywordsCriticasAusentes, pontosEliminatorios somente quando houver, e o "
    "veredicto em no maximo duas linhas. Em seguida, escreva uma linha contendo "
    "somente [[NARRACAO_ATS_ETAPA_3]] e continue com a segunda mensagem, 'Etapa 3 "
    "— Score pos-geracao', comparando o score final ao inicial e dizendo o que "
    "mudou. O marcador e interno e jamais pode aparecer ao candidato. Para texto "
    "visivel ao candidato, 'Etapa 1', 'Etapa 2' e 'Etapa 3' significam somente a "
    "metodologia ATS: Analise, Reescrita e Score pos-geracao. "
    "Se o status ainda nao for terminal, informe que a geracao esta em andamento; "
    "nao invente outra acao. Use status_geracao para verificar geracao em andamento; nunca crie "
    "definir_proximo_passo com titulo de verificar status. Ao chamar "
    "definir_proximo_passo, o args.tipo deve ser exatamente um destes enums: "
    "REVISAR_VAGA, GERAR_CURRICULO, ENVIAR_CANDIDATURA, FAZER_FOLLOW_UP, "
    "PREPARAR_ENTREVISTA, PARTICIPAR_ENTREVISTA, ENVIAR_MATERIAL, OUTRO. Nunca "
    "use texto livre em campo descrito como enum no catalogo. Para preparar texto "
    "ao recrutador, chame redigir_mensagem_recrutador; definir_proximo_passo serve "
    "somente para criar uma acao de agenda. "
    "Curriculos devem preservar fatos verdadeiros, experiencias densas, autoria de "
    "time quando aplicavel, bullets com verbo de acao, keywords honestas e pagina "
    "unica quando possivel. Se o candidato disser que atualizou o perfil ou as "
    "competencias e quer tentar novamente para a oportunidade em foco, leia o "
    "perfil e gere uma nova versao a partir dele. Nao peca Markdown nem escolha "
    "edicao de curriculo nesse caso. No texto visivel ao candidato, use apenas "
    "linguagem de produto. Nunca cite identificadores de ferramentas, rotas, "
    "payloads, JSON ou instrucoes internas. Fale em portugues, no escopo do "
    "candidato. "
    "Responda SEMPRE em JSON no formato "
    '{"tipo":"texto"|"tool_call","texto":"...","tool":"...","args":{...}}. '
    "Use tipo texto quando for so conversar e tipo tool_call quando acionar uma tool."
)

_FERRAMENTAS_INTERNAS = (
    "listar_oportunidades|buscar_oportunidade|abrir_workspace|ler_timeline|"
    "listar_acoes|ler_perfil|listar_curriculos|buscar_curriculo|status_geracao|"
    "listar_banco_vagas|ler_agenda|registrar_oportunidade|ativar_entrada|"
    "ativar_banco_vaga|gerar_curriculo|editar_curriculo|definir_proximo_passo|"
    "concluir_passo|mover_estagio|registrar_candidatura|atualizar_candidatura|"
    "registrar_nota|redigir_mensagem_recrutador|redigir_respostas_formulario"
)
_DETALHE_INTERNO = re.compile(
    rf"\b(?:{_FERRAMENTAS_INTERNAS})\b|\b(?:GET|POST|PUT|PATCH)\s+/\S+|\b(?:payload|json|tool|tools|rota)\b",
    re.IGNORECASE,
)


def _normalizar_intencao(texto: str) -> str:
    return "".join(
        caractere
        for caractere in unicodedata.normalize("NFD", texto.lower())
        if not unicodedata.combining(caractere)
    )


def _mencoes_proximas(texto: str, grupos: tuple[tuple[str, ...], tuple[str, ...]]) -> bool:
    esquerda, direita = grupos
    for a in esquerda:
        for b in direita:
            for primeiro in re.finditer(rf"\b{a}\w*\b", texto):
                if re.search(rf"\b{b}\w*\b", texto[max(0, primeiro.start() - 48) : primeiro.end() + 48]):
                    return True
    return False


def _regerar_por_perfil_atualizado(req: TurnRequest) -> TurnResponse | None:
    if not req.oportunidade_id or not req.mensagens:
        return None
    ultimas_mensagens = [_normalizar_intencao(m.conteudo) for m in req.mensagens if m.papel == "user"][-2:]
    contexto = " ".join(ultimas_mensagens)
    atualizou = bool(re.search(r"\b(atualiz|adicionei|inclui|coloquei)\w*\b", contexto))
    perfil = _mencoes_proximas(
        contexto,
        (("atualiz", "adicionei", "inclui", "coloquei"), ("perfil", "competenc")),
    )
    tentar = bool(re.search(r"\b(tente|novamente|nova versao|reger|tentar)\w*\b", contexto))
    if not (atualizou and perfil and tentar):
        return None
    ultima = req.mensagens[-1]
    if ultima.papel == "tool" and ultima.tool == "ler_perfil":
        return TurnResponse(
            tipo="tool_call",
            tool="gerar_curriculo",
            args={"oportunidadeId": req.oportunidade_id},
        )
    return TurnResponse(tipo="tool_call", tool="ler_perfil")


def _texto_para_candidato(texto: str, req: TurnRequest | None = None) -> str:
    protegido = _DETALHE_INTERNO.sub("esta acao", texto)
    if req:
        nomes = [re.escape(tool.nome) for tool in req.tools]
        if nomes:
            protegido = re.sub(rf"\b(?:{'|'.join(nomes)})\b", "esta acao", protegido)
    return protegido


def _narracao_ats_concluida(req: TurnRequest) -> TurnResponse | None:
    ultima = req.mensagens[-1] if req.mensagens else None
    if not ultima or ultima.papel != "tool" or ultima.tool != "buscar_curriculo":
        return None
    try:
        curriculo = json.loads(ultima.conteudo)
    except json.JSONDecodeError:
        return None
    if not isinstance(curriculo, dict):
        return None
    inicial = curriculo.get("analiseInicial")
    final = curriculo.get("analiseFinal")
    if not isinstance(inicial, dict) or not isinstance(final, dict):
        return None
    score_inicial = inicial.get("score")
    score_final = final.get("score")
    if not isinstance(score_inicial, (int, float)) or not isinstance(score_final, (int, float)):
        return None

    def lista(campo: str) -> str:
        valores = inicial.get(campo)
        if not isinstance(valores, list):
            return "Nenhuma"
        itens = [str(valor).strip() for valor in valores if str(valor).strip()]
        return ", ".join(itens) if itens else "Nenhuma"

    pontos = lista("pontosEliminatorios")
    linhas_iniciais = [
        "Etapa 1 — Analise ATS",
        f"Score: {score_inicial}",
        f"Keywords encontradas: {lista('keywordsEncontradas')}",
        f"Keywords criticas ausentes: {lista('keywordsCriticasAusentes')}",
    ]
    if pontos != "Nenhuma":
        linhas_iniciais.append(f"Pontos eliminatorios: {pontos}")
    linhas_iniciais.append(f"Veredicto: {str(inicial.get('veredicto') or 'Sem veredicto informado.')}")

    diferenca = score_final - score_inicial
    if diferenca > 0:
        comparacao = f"aumentou {diferenca:g} ponto(s)"
    elif diferenca < 0:
        comparacao = f"reduziu {abs(diferenca):g} ponto(s)"
    else:
        comparacao = "permaneceu igual"
    texto = (
        "\n".join(linhas_iniciais)
        + "\n\n[[NARRACAO_ATS_ETAPA_3]]\n\n"
        + "Etapa 3 — Score pos-geracao\n"
        + f"Score final: {score_final}, comparado ao inicial de {score_inicial}: {comparacao}."
    )
    return TurnResponse(tipo="texto", texto=texto)


def _catalogo(req: TurnRequest) -> str:
    linhas = []
    for t in req.tools:
        params = (
            ", ".join(f"{nome}: {regra}" for nome, regra in t.parametros.items())
            if t.parametros
            else "sem parametros"
        )
        linhas.append(f"- {t.nome} [{t.efeito}]: {t.descricao} | args: {params}")
    return "\n".join(linhas)


def _conversa(req: TurnRequest) -> str:
    linhas = []
    for m in req.mensagens:
        rotulo = m.tool or m.papel
        linhas.append(f"[{rotulo}] {m.conteudo}")
    return "\n".join(linhas)


def _user(req: TurnRequest) -> str:
    alvo = req.oportunidade_id or "nenhuma"
    return (
        f"Modo: {req.modo}. Oportunidade em foco: {alvo}.\n\n"
        f"Tools disponiveis:\n{_catalogo(req)}\n\n"
        f"Conversa ate aqui:\n{_conversa(req)}\n\n"
        "Decida o proximo passo e responda no formato JSON pedido."
    )


def planejar_turno(req: TurnRequest) -> TurnResponse:
    regeracao = _regerar_por_perfil_atualizado(req)
    if regeracao:
        return regeracao
    narracao = _narracao_ats_concluida(req)
    if narracao:
        return narracao
    try:
        res = complete_model(SYSTEM_TURNO, _user(req), TurnResponse)
        if res.tipo in ("texto", "tool_call"):
            if res.tipo == "tool_call" and not res.tool:
                return TurnResponse(tipo="texto", texto=_texto_para_candidato(res.texto or "", req))
            if res.tipo == "texto":
                return TurnResponse(tipo="texto", texto=_texto_para_candidato(res.texto or "", req))
            return res
    except LLMUnavailable:
        raise


SYSTEM_MENSAGEM = (
    "Voce redige uma mensagem curta e profissional do candidato para o recrutador "
    "da vaga. Use apenas fatos do perfil do candidato. Tom cordial e objetivo, sem "
    "exageros. O candidato vai revisar e enviar. Responda em JSON."
)


def _perfil_txt(perfil: PerfilMestre) -> str:
    partes = [perfil.nome, perfil.resumo]
    if perfil.contato:
        partes.append("Contato: " + " | ".join(perfil.contato.values()))
    if perfil.experiencias:
        experiencias = "\n\n".join(experiencia.texto for experiencia in perfil.experiencias if experiencia.texto)
        if experiencias:
            partes.append("Experiências:\n" + experiencias)
    if perfil.skills:
        partes.append("Skills: " + ", ".join(perfil.skills))
    return "\n".join(p for p in partes if p)


def redigir_mensagem(req: RedigirMensagemRequest) -> RedigirMensagemResponse:
    user = (
        f"Vaga: {req.vaga.titulo} @ {req.vaga.empresa}\n"
        f"Descricao:\n{req.vaga.descricao}\n\n"
        f"Candidato:\n{_perfil_txt(req.perfil)}\n\n"
        f"Contexto adicional: {req.contexto}\n\n"
        'Devolva JSON {"titulo":"...","texto":"...","destino":"..."} com a mensagem '
        "pronta para copiar."
    )
    try:
        res = complete_model(SYSTEM_MENSAGEM, user, RedigirMensagemResponse)
        if res.texto.strip():
            return res
    except LLMUnavailable:
        raise


SYSTEM_FORMULARIO = (
    "Voce redige respostas do candidato para campos de um formulario de "
    "candidatura. Cada resposta e curta, verdadeira ao perfil e alinhada a vaga. O "
    "candidato revisa antes de usar. Responda em JSON."
)


def redigir_formulario(req: RedigirFormularioRequest) -> RedigirFormularioResponse:
    campos = "\n".join(f"- {c}" for c in req.campos)
    user = (
        f"Vaga: {req.vaga.titulo} @ {req.vaga.empresa}\n"
        f"Descricao:\n{req.vaga.descricao}\n\n"
        f"Candidato:\n{_perfil_txt(req.perfil)}\n\n"
        f"Campos do formulario:\n{campos}\n\n"
        'Devolva JSON {"titulo":"...","respostas":[{"campo":"...","texto":"..."}],'
        '"texto":"..."} com uma resposta por campo e um texto consolidado.'
    )
    try:
        res = complete_model(SYSTEM_FORMULARIO, user, RedigirFormularioResponse)
        if res.respostas:
            return res
    except LLMUnavailable:
        raise
