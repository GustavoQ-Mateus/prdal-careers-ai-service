import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from app.casamento import termo_presente
from app.fontes import fontes_da_geracao
from app.generate import TETO_REQUISICOES, DiagnosticoGeracao, generate_cv_pipeline
from app.orcamento import BULLETS_TOTAL, COMPETENCIAS_MAX_CATEGORIAS, COMPETENCIAS_MAX_TERMOS, RESUMO_MAX_FRASES, bullets_maximos
from app.renderizador import experiencias_por_recencia
from app.schemas import EstruturaCurriculo, GenerateCvRequest
from app.verificacao import _metricas_sem_fonte

from ..falso import cliente_falso
from ..nucleo import Uso, ler_json, media, medir
from .juiz import FraseJulgada, julgar

NOME = "geracao"
PASTA = Path(__file__).resolve().parent
TRAVESSOES = ("\u2014", "\u2013")
MARCAS_INTERNAS = ("{{", "}}", "[[", "]]")
_SECAO_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def carregar() -> list[dict[str, Any]]:
    return ler_json(PASTA / "casos.json")


def _roteiros() -> dict[str, list[dict[str, Any]]]:
    return ler_json(PASTA / "falso.json")


def requisicao(caso: dict[str, Any]) -> GenerateCvRequest:
    return GenerateCvRequest.model_validate(
        {
            "perfilMestre": caso["perfil"],
            "vaga": caso["vaga"],
            "keywords": caso["keywords"],
            "contexto": caso.get("contexto", []),
        }
    )


def frases_da_estrutura(estrutura: EstruturaCurriculo) -> list[tuple[str, str, list[str]]]:
    frases: list[tuple[str, str, list[str]]] = []
    if estrutura.titulo and estrutura.titulo.texto.strip():
        frases.append(("titulo", estrutura.titulo.texto, list(estrutura.titulo.fontes)))
    for indice, frase in enumerate(estrutura.resumo):
        frases.append((f"resumo.{indice + 1}", frase.texto, list(frase.fontes)))
    for experiencia in estrutura.experiencias:
        if experiencia.experiencia_id in estrutura.experiencias_omitidas:
            continue
        for indice, frase in enumerate(experiencia.bullets):
            frases.append((f"bullet.{experiencia.experiencia_id}.{indice + 1}", frase.texto, list(frase.fontes)))
    return [frase for frase in frases if frase[1].strip()]


def metricas_sem_fonte(frases: list[tuple[str, str, list[str]]], textos: dict[str, str]) -> list[dict[str, Any]]:
    achados = []
    for chave, texto, fontes in frases:
        base = "\n".join(textos.get(fonte_id, "") for fonte_id in fontes)
        faltando = _metricas_sem_fonte(texto, base)
        if faltando:
            achados.append({"chave": chave, "metricas": faltando})
    return achados


def violacoes_de_formato(markdown: str, estrutura: EstruturaCurriculo, req: GenerateCvRequest) -> list[str]:
    violacoes: list[str] = []
    if not markdown.startswith("# "):
        violacoes.append("curriculo nao comeca com o nome em titulo")
    violacoes += [f"travessao no texto: {repr(c)}" for c in TRAVESSOES if c in markdown]
    violacoes += [f"marca interna no texto: {m}" for m in MARCAS_INTERNAS if m in markdown]
    secoes = _SECAO_RE.findall(markdown)
    violacoes += [f"secao repetida: {s}" for s in sorted({s for s in secoes if secoes.count(s) > 1})]
    blocos = re.split(r"^## .+$", markdown, flags=re.MULTILINE)[1:]
    violacoes += [f"secao vazia: {s}" for s, corpo in zip(secoes, blocos) if not corpo.strip()]
    if len(estrutura.resumo) > RESUMO_MAX_FRASES:
        violacoes.append(f"resumo com {len(estrutura.resumo)} frases")
    bullets = {item.experiencia_id: item.bullets for item in estrutura.experiencias}
    total = 0
    for posicao, (experiencia_id, _) in enumerate(experiencias_por_recencia(req.perfil_mestre)):
        if experiencia_id in estrutura.experiencias_omitidas:
            continue
        quantidade = len(bullets.get(experiencia_id, []))
        total += quantidade
        if quantidade > bullets_maximos(posicao):
            violacoes.append(f"experiencia {experiencia_id} com {quantidade} bullets")
    conhecidas = {experiencia_id for experiencia_id, _ in experiencias_por_recencia(req.perfil_mestre)}
    violacoes += [f"experiencia desconhecida: {i}" for i in bullets if i not in conhecidas]
    if total > BULLETS_TOTAL:
        violacoes.append(f"{total} bullets no total")
    if len(estrutura.competencias) > COMPETENCIAS_MAX_CATEGORIAS:
        violacoes.append(f"{len(estrutura.competencias)} categorias de competencias")
    termos = sum(len(categoria.termos) for categoria in estrutura.competencias)
    if termos > COMPETENCIAS_MAX_TERMOS:
        violacoes.append(f"{termos} termos de competencias")
    return violacoes


def _frases_do_juiz(caso: dict[str, Any], frases: list[tuple[str, str, list[str]]], textos: dict[str, str]) -> list[FraseJulgada]:
    julgadas = [
        FraseJulgada(chave, texto, tuple((fonte_id, textos.get(fonte_id, "")) for fonte_id in fontes))
        for chave, texto, fontes in frases
    ]
    for indice, sonda in enumerate(caso.get("sondasJuiz", [])):
        fontes = tuple((fonte_id, textos.get(fonte_id, "")) for fonte_id in sonda["fontes"])
        julgadas.append(FraseJulgada(f"sonda.{indice + 1}", sonda["texto"], fontes))
    return julgadas


def rodar_caso(caso: dict[str, Any], falso: bool) -> dict[str, Any]:
    req = requisicao(caso)
    textos = {fonte.id: fonte.texto for fonte in fontes_da_geracao(req).values()}
    uso_geracao = Uso()
    uso_juiz = Uso()
    contexto = cliente_falso(_roteiros()[caso["id"]]) if falso else nullcontext()
    with contexto:
        diagnostico = DiagnosticoGeracao()
        with medir(uso_geracao):
            resposta = generate_cv_pipeline(req, diagnostico)
        estrutura = resposta.estrutura or EstruturaCurriculo()
        frases = frases_da_estrutura(estrutura)
        with medir(uso_juiz):
            notas = julgar(req.vaga.descricao, req.keywords, _frases_do_juiz(caso, frases, textos))

    sem_fonte = metricas_sem_fonte(frases, textos)
    proibidos = [termo for termo in caso.get("termosProibidos", []) if termo_presente(termo, resposta.markdown)]
    formato = violacoes_de_formato(resposta.markdown, estrutura, req)
    keywords = [k.termo for k in req.keywords if k.termo.strip()]
    reais = [nota for chave, nota in notas.items() if not chave.startswith("sonda.")]
    sondas = [
        {
            "chave": f"sonda.{indice + 1}",
            "texto": sonda["texto"],
            "esperado": sonda["relacaoSustentada"],
            "juiz": notas[f"sonda.{indice + 1}"].relacao_sustentada if f"sonda.{indice + 1}" in notas else None,
        }
        for indice, sonda in enumerate(caso.get("sondasJuiz", []))
    ]
    uso = Uso()
    uso.somar(uso_geracao)
    uso.somar(uso_juiz)
    metricas = {
        "frases_rejeitadas": len(diagnostico.rejeitadas),
        "frases_reparadas": diagnostico.reparadas,
        "frases_descartadas": len(diagnostico.descartadas),
        "metrica_sem_fonte": len(sem_fonte),
        "termos_proibidos_presentes": len(proibidos),
        "cobertura_keywords": round(len(resposta.analise_final.keywords_encontradas) / len(keywords), 4) if keywords else None,
        "requisicoes": uso_geracao.requisicoes,
        "violacoes_formato": len(formato),
        "degradada": 1 if resposta.degradacao else 0,
        "juiz_requisito": media([1.0 if n.responde_requisito else 0.0 for n in reais]),
        "juiz_relacao": media([1.0 if n.relacao_sustentada else 0.0 for n in reais]),
        "juiz_sondas": media([1.0 if s["juiz"] == s["esperado"] else 0.0 for s in sondas]) if sondas else None,
    }
    return {
        "id": caso["id"],
        "mede": caso["mede"],
        "metricas": metricas,
        "uso": uso,
        "detalhes": {
            "usoGeracao": uso_geracao.como_dict(),
            "usoJuiz": uso_juiz.como_dict(),
            "score": {"inicial": resposta.analise_inicial.score, "final": resposta.analise_final.score},
            "keywordsEncontradas": resposta.analise_final.keywords_encontradas,
            "rejeitadas": [{"chave": r.chave, "texto": r.texto, "motivo": r.motivo} for r in diagnostico.rejeitadas],
            "descartadas": [{"chave": r.chave, "texto": r.texto, "motivo": r.motivo} for r in diagnostico.descartadas],
            "metricasSemFonte": sem_fonte,
            "termosProibidosPresentes": proibidos,
            "violacoesFormato": formato,
            "degradacao": resposta.degradacao,
            "notasDoJuiz": [
                {"chave": chave, "texto": texto, "fontes": fontes, **_nota(notas.get(chave))}
                for chave, texto, fontes in frases
            ],
            "sondasDoJuiz": sondas,
            "markdown": resposta.markdown,
        },
    }


def _nota(nota: Any) -> dict[str, Any]:
    if nota is None:
        return {"respondeRequisito": None, "requisitoCitado": "", "relacaoSustentada": None, "justificativa": "sem nota"}
    return nota.model_dump(by_alias=True, exclude={"chave"})


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    def valores(nome: str) -> list[Any]:
        return [r["metricas"][nome] for r in resultados if r["metricas"].get(nome) is not None]

    def soma(nome: str) -> int:
        return sum(valores(nome))

    notas = [n for r in resultados for n in r["detalhes"]["notasDoJuiz"] if n["relacaoSustentada"] is not None]
    sondas = [s for r in resultados for s in r["detalhes"]["sondasDoJuiz"]]
    requisicoes = valores("requisicoes")
    return {
        "metrica_sem_fonte": soma("metrica_sem_fonte"),
        "termos_proibidos_presentes": soma("termos_proibidos_presentes"),
        "violacoes_formato": soma("violacoes_formato"),
        "degradadas": soma("degradada"),
        "cobertura_keywords": media(valores("cobertura_keywords")),
        "requisicoes_media": media(requisicoes),
        "requisicoes_max": max(requisicoes) if requisicoes else None,
        "frases_rejeitadas": soma("frases_rejeitadas"),
        "frases_reparadas": soma("frases_reparadas"),
        "frases_descartadas": soma("frases_descartadas"),
        "juiz_requisito": media([1.0 if n["respondeRequisito"] else 0.0 for n in notas]),
        "juiz_relacao": media([1.0 if n["relacaoSustentada"] else 0.0 for n in notas]),
        "juiz_sondas": media([1.0 if s["juiz"] == s["esperado"] else 0.0 for s in sondas]) if sondas else None,
    }


def extras(_resultados: list[dict[str, Any]]) -> dict[str, Any]:
    return {"tetoDeRequisicoesPorGeracao": TETO_REQUISICOES, "juizEmProducao": False}
