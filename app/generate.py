from typing import Any

from .llm import LLMUnavailable, complete_model
from .schemas import ExperienciaPerfil, GenerateCvRequest, GenerateCvResponse, PerfilMestre

SYSTEM = (
    "Voce escreve curriculos em Markdown otimizados para ATS. Use apenas fatos "
    "do perfil-mestre fornecido, nunca invente experiencias. Espelhe as "
    "palavras-chave da vaga quando forem verdadeiras para o candidato. Estrutura "
    "com secoes Resumo, Experiencia, Formacao, Skills e Contato. Responda em JSON."
)


def _user(req: GenerateCvRequest) -> str:
    termos = ", ".join(k.termo for k in req.keywords)
    return (
        "Gere o Markdown do curriculo e devolva JSON no formato "
        '{"markdown":"..."}.\n\n'
        f"Perfil-mestre:\n{req.perfil_mestre.model_dump_json()}\n\n"
        f"Vaga: {req.vaga.titulo} @ {req.vaga.empresa}\n"
        f"Descricao da vaga:\n{req.vaga.descricao}\n\n"
        f"Palavras-chave a priorizar: {termos}\n\n"
        f"Contexto adicional:\n{chr(10).join(req.contexto)}"
    )


def _linha_contato(contato: dict[str, Any]) -> str:
    partes = [str(v) for v in contato.values() if v]
    return " | ".join(partes)


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
    linhas = [f"# {perfil.nome}".strip()]
    contato = _linha_contato(perfil.contato)
    if contato:
        linhas.append(contato)
    if perfil.resumo:
        linhas += ["", "## Resumo", perfil.resumo]
    if perfil.experiencias:
        linhas += ["", "## Experiencia"]
        for experiencia in perfil.experiencias:
            texto = _texto_experiencia(experiencia)
            if texto:
                linhas += ["", f"### {_titulo_experiencia(experiencia)}", texto]
    if perfil.formacao:
        linhas += ["", "## Formacao", *[f"- {formacao}" for formacao in perfil.formacao]]
    if perfil.skills:
        linhas += ["", "## Skills", ", ".join(perfil.skills)]
    return "\n".join(linhas).strip() or "# Curriculo"


def generate_cv(req: GenerateCvRequest) -> str:
    try:
        res = complete_model(SYSTEM, _user(req), GenerateCvResponse)
        if res.markdown.strip():
            return res.markdown
    except LLMUnavailable:
        pass
    return _deterministic(req.perfil_mestre)
