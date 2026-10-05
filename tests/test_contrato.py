import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.contrato import conferir_contrato, gerar_contrato

RAIZ = Path(__file__).resolve().parents[1]


def test_contrato_coincide_com_o_codigo():
    conferir_contrato(RAIZ / "contrato" / "openapi.json")
    documento = json.loads(gerar_contrato())
    assert "/keywords" in documento["paths"]
    assert "KeywordsResponse" in documento["components"]["schemas"]


def test_contrato_divergente_falha(tmp_path):
    caminho = tmp_path / "openapi.json"
    caminho.write_text(gerar_contrato().replace('"ai-service"', '"divergente"'), encoding="utf-8")
    with pytest.raises(ValueError, match="contrato do ai-service desatualizado"):
        conferir_contrato(caminho)


def test_exporta_em_producao_sem_rotas_de_documentacao(tmp_path):
    caminho = tmp_path / "openapi.json"
    codigo = "from app.main import app; assert app.docs_url is None; assert app.openapi_url is None; from scripts.contrato import executar; executar()"
    resultado = subprocess.run(
        [sys.executable, "-c", codigo, "--destino", str(caminho)],
        cwd=RAIZ,
        env={**os.environ, "PRDAL_AMBIENTE": "producao"},
        capture_output=True,
        text=True,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert caminho.read_text(encoding="utf-8") == gerar_contrato()
