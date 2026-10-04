import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Chunk, IngestResponse, QueryResponse
from app.seguranca import HEADER_SERVICO, TokenServicoAusente, exigir_token_no_boot

TOKEN = "token-de-servico-de-teste-com-mais-de-32-bytes"
PRODUCAO = {"PRDAL_AMBIENTE": "producao", "SERVICE_TOKEN": TOKEN}
RAIZ = Path(__file__).resolve().parents[1]

CONSULTA = {"usuarioId": "usuario-vitima", "query": "pretensao salarial", "k": 5}
SUBSTITUICAO = {"usuarioId": "usuario-vitima", "documentos": []}


def _consulta_falsa(*_args, **_kwargs):
    return QueryResponse(chunks=[Chunk(id="n1", tipo="nota", texto="nota privada", origem="nota", titulo="Nota")])


@patch("app.main.substituir", return_value=IngestResponse(indexados=0))
@patch("app.main.consultar", side_effect=_consulta_falsa)
@patch.dict(os.environ, PRODUCAO)
class ServicoAutenticadoTest(unittest.TestCase):
    def setUp(self):
        self.cliente = TestClient(app)

    def test_context_query_sem_header_retorna_401_e_nao_consulta(self, consultar, _substituir):
        resposta = self.cliente.post("/context/query", json=CONSULTA)
        self.assertEqual(resposta.status_code, 401)
        consultar.assert_not_called()

    def test_context_replace_sem_header_retorna_401_e_nao_apaga(self, _consultar, substituir):
        resposta = self.cliente.post("/context/replace", json=SUBSTITUICAO)
        self.assertEqual(resposta.status_code, 401)
        substituir.assert_not_called()

    def test_token_errado_retorna_401(self, consultar, _substituir):
        for errado in ("", "x", TOKEN[:-1], TOKEN + "x", TOKEN.upper()):
            resposta = self.cliente.post("/context/query", json=CONSULTA, headers={HEADER_SERVICO: errado})
            self.assertEqual(resposta.status_code, 401, errado)
        consultar.assert_not_called()

    def test_com_header_correto_funciona(self, consultar, substituir):
        cabecalho = {HEADER_SERVICO: TOKEN}
        consulta = self.cliente.post("/context/query", json=CONSULTA, headers=cabecalho)
        self.assertEqual(consulta.status_code, 200)
        self.assertEqual(consulta.json()["chunks"][0]["texto"], "nota privada")
        substituicao = self.cliente.post("/context/replace", json=SUBSTITUICAO, headers=cabecalho)
        self.assertEqual(substituicao.status_code, 200)
        consultar.assert_called_once()
        substituir.assert_called_once()

    def test_ingest_e_geracao_tambem_exigem_header(self, _consultar, _substituir):
        for rota in ("/context/ingest", "/generate-cv-pipeline", "/copiloto/turn", "/keywords", "/hello"):
            resposta = self.cliente.post(rota, json={}) if rota != "/hello" else self.cliente.get(rota)
            self.assertEqual(resposta.status_code, 401, rota)

    def test_health_continua_publico(self, _consultar, _substituir):
        self.assertEqual(self.cliente.get("/health").status_code, 200)


class BootSemTokenTest(unittest.TestCase):
    @patch.dict(os.environ, {"PRDAL_AMBIENTE": "producao", "SERVICE_TOKEN": ""})
    def test_fora_de_desenvolvimento_nao_sobe_sem_token(self):
        with self.assertRaises(TokenServicoAusente):
            exigir_token_no_boot()
        with self.assertRaises(TokenServicoAusente):
            with TestClient(app):
                pass

    @patch.dict(os.environ, {"PRDAL_AMBIENTE": "producao", "SERVICE_TOKEN": "curto"})
    def test_token_curto_e_recusado(self):
        with self.assertRaises(TokenServicoAusente):
            exigir_token_no_boot()

    @patch.dict(os.environ, {"SERVICE_TOKEN": ""})
    def test_sem_ambiente_definido_vale_como_producao(self):
        os.environ.pop("PRDAL_AMBIENTE", None)
        with self.assertRaises(TokenServicoAusente):
            exigir_token_no_boot()
        self.assertEqual(TestClient(app).post("/context/query", json=CONSULTA).status_code, 401)

    @patch.dict(os.environ, {"PRDAL_AMBIENTE": "desenvolvimento", "SERVICE_TOKEN": ""})
    def test_em_desenvolvimento_sobe_sem_token(self):
        exigir_token_no_boot()


class DocumentacaoForaDeDesenvolvimentoTest(unittest.TestCase):
    def _rotas(self, ambiente: str) -> dict:
        codigo = (
            "import json\n"
            "from fastapi.testclient import TestClient\n"
            "from app.main import app\n"
            f"c = TestClient(app, headers={{'{HEADER_SERVICO}': '{TOKEN}'}})\n"
            "print(json.dumps({r: c.get(r).status_code for r in ('/docs', '/redoc', '/openapi.json')}))\n"
        )
        env = {**os.environ, "PRDAL_AMBIENTE": ambiente, "SERVICE_TOKEN": TOKEN}
        saida = subprocess.run(
            [sys.executable, "-c", codigo], cwd=RAIZ, env=env, capture_output=True, text=True, check=True
        )
        return json.loads(saida.stdout.strip().splitlines()[-1])

    def test_producao_nao_publica_docs_nem_openapi(self):
        self.assertEqual(self._rotas("producao"), {"/docs": 404, "/redoc": 404, "/openapi.json": 404})

    def test_desenvolvimento_mantem_docs(self):
        self.assertEqual(self._rotas("desenvolvimento"), {"/docs": 200, "/redoc": 200, "/openapi.json": 200})


if __name__ == "__main__":
    unittest.main()
