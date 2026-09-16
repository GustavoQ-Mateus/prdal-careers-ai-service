import json
import re

from .llm import LLMUnavailable, complete_model
from .schemas import (
    PerfilMestre,
    RedigirFormularioRequest,
    RedigirFormularioResponse,
    RedigirMensagemRequest,
    RedigirMensagemResponse,
    RespostaFormulario,
    TurnRequest,
    TurnResponse,
    Vaga,
)

SYSTEM_TURNO = (
    "Voce e o copiloto de candidatura, um agente que conduz o candidato pela "
    "preparacao de vagas. Raciocina e escolhe o proximo passo, mas quem executa "
    "as tools e a api; voce so declara a intencao. Use apenas as tools listadas. "
    "Aja uma tool por vez. Peca leitura antes de escrita. Nunca invente numero de "
    "score; o score vem sempre da tool. Nunca envie nada externo por conta propria; "
    "para mensagem a recrutador ou resposta de formulario, use as tools de redacao "
    "que entregam texto ao candidato revisar. Fale em portugues, no escopo do "
    "candidato. Responda SEMPRE em JSON no formato "
    '{"tipo":"texto"|"tool_call","texto":"...","tool":"...","args":{...}}. '
    "Use tipo texto quando for so conversar e tipo tool_call quando acionar uma tool."
)

_DIRETIVA = re.compile(r"^\s*tool\s+([a-z_]+)\s*(\{.*\})?\s*$", re.IGNORECASE | re.DOTALL)


def _catalogo(req: TurnRequest) -> str:
    linhas = []
    for t in req.tools:
        params = ", ".join(t.parametros.keys()) if t.parametros else "sem parametros"
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


def _fallback(req: TurnRequest) -> TurnResponse:
    ultima = req.mensagens[-1] if req.mensagens else None
    if ultima and ultima.papel == "user":
        m = _DIRETIVA.match(ultima.conteudo)
        if m:
            nome = m.group(1).lower()
            nomes = {t.nome for t in req.tools}
            if nome in nomes:
                args = {}
                if m.group(2):
                    try:
                        args = json.loads(m.group(2))
                    except json.JSONDecodeError:
                        args = {}
                return TurnResponse(tipo="tool_call", tool=nome, args=args)
        return TurnResponse(
            tipo="texto",
            texto=(
                "Modelo indisponivel para raciocinar. Posso seguir com passos "
                "explicitos: envie tool <nome> com os argumentos."
            ),
        )
    return TurnResponse(tipo="texto", texto="Etapa concluida.")


def planejar_turno(req: TurnRequest) -> TurnResponse:
    try:
        res = complete_model(SYSTEM_TURNO, _user(req), TurnResponse)
        if res.tipo in ("texto", "tool_call"):
            if res.tipo == "tool_call" and not res.tool:
                return TurnResponse(tipo="texto", texto=res.texto or "")
            return res
    except LLMUnavailable:
        pass
    return _fallback(req)


SYSTEM_MENSAGEM = (
    "Voce redige uma mensagem curta e profissional do candidato para o recrutador "
    "da vaga. Use apenas fatos do perfil do candidato. Tom cordial e objetivo, sem "
    "exageros. O candidato vai revisar e enviar. Responda em JSON."
)


def _perfil_txt(perfil: PerfilMestre) -> str:
    partes = [perfil.nome, perfil.resumo]
    if perfil.skills:
        partes.append("Skills: " + ", ".join(str(s) for s in perfil.skills))
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
        pass
    nome = req.perfil.nome or "o candidato"
    texto = (
        f"Ola, tenho interesse na vaga de {req.vaga.titulo} na {req.vaga.empresa}. "
        f"{req.perfil.resumo} Fico a disposicao para conversar. Obrigado."
    ).strip()
    return RedigirMensagemResponse(
        titulo=f"Mensagem ao recrutador de {req.vaga.titulo}",
        texto=texto,
        destino=f"recrutador de {req.vaga.empresa}",
    )


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
        pass
    respostas = [
        RespostaFormulario(
            campo=c,
            texto=f"{req.perfil.resumo}".strip() or "Resposta a revisar.",
        )
        for c in req.campos
    ]
    texto = "\n\n".join(f"{r.campo}\n{r.texto}" for r in respostas)
    return RedigirFormularioResponse(
        titulo=f"Respostas para {req.vaga.titulo}",
        respostas=respostas,
        texto=texto,
    )
