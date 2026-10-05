import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.main import app


def gerar_contrato() -> str:
    app.openapi_schema = None
    return json.dumps(app.openapi(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def conferir_contrato(caminho: Path) -> None:
    if caminho.read_text(encoding="utf-8") != gerar_contrato():
        raise ValueError("contrato do ai-service desatualizado")


def executar() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conferir", action="store_true")
    parser.add_argument("--destino", type=Path, default=RAIZ / "contrato" / "openapi.json")
    args = parser.parse_args()
    if args.conferir:
        conferir_contrato(args.destino)
    else:
        args.destino.parent.mkdir(parents=True, exist_ok=True)
        args.destino.write_text(gerar_contrato(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    executar()
