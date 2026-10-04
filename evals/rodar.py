import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from .nucleo import (
    RAIZ,
    ConfiguracaoAusente,
    Uso,
    comparar,
    exigir_credencial,
    gravar_json,
    ler_json,
    precos_mtok,
    regras_atualizadas,
)

CONJUNTOS = {
    "geracao": "evals.geracao.conjunto",
    "agente": "evals.agente.conjunto",
    "rag": "evals.rag.conjunto",
}
LINHA_DE_BASE = RAIZ / "linha_de_base.json"
LINHA_DE_BASE_FALSO = RAIZ / "linha_de_base_falso.json"


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.rodar", description="Roda um conjunto de avaliacao")
    parser.add_argument("--conjunto", required=True, choices=sorted(CONJUNTOS))
    parser.add_argument("--falso", action="store_true", help="usa respostas roteiradas, sem chamar o modelo")
    parser.add_argument("--saida", help="diretorio dos relatorios")
    parser.add_argument("--linha-de-base", help="arquivo da linha de base")
    parser.add_argument("--gravar-linha-de-base", action="store_true", help="grava o resultado como linha de base")
    parser.add_argument("--caso", action="append", default=[], help="roda so os casos com este id")
    return parser.parse_args(argv)


def _rodar_caso(modulo: Any, caso: dict[str, Any], falso: bool) -> dict[str, Any]:
    inicio = time.perf_counter()
    try:
        resultado = modulo.rodar_caso(caso, falso)
    except Exception as exc:
        resultado = {"id": caso["id"], "erro": f"{type(exc).__name__}: {exc}", "metricas": {}, "uso": Uso()}
    resultado["duracaoMs"] = round((time.perf_counter() - inicio) * 1000)
    return resultado


def _serializavel(resultado: dict[str, Any]) -> dict[str, Any]:
    uso = resultado.get("uso")
    return {**resultado, "uso": uso.como_dict() if isinstance(uso, Uso) else uso}


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    modulo = importlib.import_module(CONJUNTOS[args.conjunto])
    if not args.falso and getattr(modulo, "EXIGE_CREDENCIAL", True):
        try:
            exigir_credencial()
        except ConfiguracaoAusente as exc:
            print(f"conjunto {args.conjunto} nao rodou: {exc}", file=sys.stderr)
            return 2

    casos = modulo.carregar()
    if args.caso:
        casos = [caso for caso in casos if caso["id"] in args.caso]
    marca = time.strftime("%Y%m%d-%H%M%S") + ("-falso" if args.falso else "")
    saida = Path(args.saida) if args.saida else RAIZ / "relatorios" / args.conjunto / marca

    resultados = []
    uso_total = Uso()
    for caso in casos:
        resultado = _rodar_caso(modulo, caso, args.falso)
        resultados.append(resultado)
        if isinstance(resultado.get("uso"), Uso):
            uso_total.somar(resultado["uso"])
        gravar_json(saida / f"{caso['id']}.json", _serializavel(resultado))
        situacao = f"erro: {resultado['erro']}" if resultado.get("erro") else "ok"
        print(f"[{args.conjunto}] {caso['id']}: {situacao}")

    resumo = modulo.resumir([r for r in resultados if not r.get("erro")])
    caminho_base = Path(args.linha_de_base) if args.linha_de_base else (LINHA_DE_BASE_FALSO if args.falso else LINHA_DE_BASE)
    base = ler_json(caminho_base)
    regras = base[args.conjunto]["metricas"]
    if args.caso or args.gravar_linha_de_base:
        regras_da_comparacao = {m: {**r, "valor": None} for m, r in regras.items()}
    else:
        regras_da_comparacao = regras
    regressoes = comparar(resumo, regras_da_comparacao)
    erros = [{"id": r["id"], "erro": r["erro"]} for r in resultados if r.get("erro")]

    relatorio = {
        "conjunto": args.conjunto,
        "modelo": "falso" if args.falso else os.getenv("AI_MODEL"),
        "falso": args.falso,
        "casos": len(resultados),
        "metricas": resumo,
        "linhaDeBase": str(caminho_base),
        "regressoes": [r.como_dict() for r in regressoes],
        "erros": erros,
        "uso": uso_total.como_dict(),
        "precosMTokUsd": precos_mtok(),
        **(modulo.extras(resultados) if hasattr(modulo, "extras") else {}),
    }
    gravar_json(saida / "resumo.json", relatorio)

    for chave, valor in resumo.items():
        print(f"  {chave}: {valor}")
    print(f"  uso: {uso_total.como_dict()}")
    print(f"  relatorios em {saida}")
    for regressao in regressoes:
        print(f"  REGRESSAO {regressao.metrica}: {regressao.motivo} (atual {regressao.atual})", file=sys.stderr)
    for erro in erros:
        print(f"  ERRO {erro['id']}: {erro['erro']}", file=sys.stderr)

    if args.gravar_linha_de_base:
        if regressoes or erros:
            print("  linha de base nao gravada: ha erro ou limite absoluto violado", file=sys.stderr)
            return 1
        base[args.conjunto]["metricas"] = regras_atualizadas(resumo, regras)
        base[args.conjunto]["gravadaEm"] = time.strftime("%Y-%m-%d")
        base[args.conjunto]["modelo"] = relatorio["modelo"]
        gravar_json(caminho_base, base)
        print(f"  linha de base gravada em {caminho_base}")
        return 0
    return 1 if regressoes or erros else 0


if __name__ == "__main__":
    sys.exit(main())
