import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.seguranca import HEADER_SERVICO, TokenServicoAusente, exigir_token_no_boot

TOKEN = "token-de-servico-de-teste-com-mais-de-32-bytes"
PRODUCAO = {"PRDAL_AMBIENTE": "producao", "SERVICE_TOKEN": TOKEN}
RAIZ = Path(__file__).resolve().parents[1]

CONSULTA = {"consultas": ["pretensao salarial"]}
DOCUMENTOS = {"documentos": [{"id": "d1", "origemId": "resumo", "tipo": "resumo", "texto": "nota privada"}]}


def _vetores_falsos(textos):
    return [[0.1] * 384 for _ in textos]


@patch("app.main.rag._vetores", side_effect=_vetores_falsos)
@patch.dict(os.environ, PRODUCAO)
class ServicoAutenticadoTest(unittest.TestCase):
    def setUp(self):
        self.cliente = TestClient(app)

    def test_embeddings_sem_header_retornam_401_e_nao_calculam(self, vetores):
        for rota, corpo in (("/embeddings/consultas", CONSULTA), ("/embeddings/documentos", DOCUMENTOS)):
            resposta = self.cliente.post(rota, json=corpo)
            self.assertEqual(resposta.status_code, 401, rota)
        vetores.assert_not_called()

    def test_token_errado_retorna_401(self, vetores):
        for errado in ("", "x", TOKEN[:-1], TOKEN + "x", TOKEN.upper()):
            resposta = self.cliente.post("/embeddings/consultas", json=CONSULTA, headers={HEADER_SERVICO: errado})
            self.assertEqual(resposta.status_code, 401, errado)
        vetores.assert_not_called()

    def test_com_header_correto_funciona(self, vetores):
        cabecalho = {HEADER_SERVICO: TOKEN}
        consulta = self.cliente.post("/embeddings/consultas", json=CONSULTA, headers=cabecalho)
        self.assertEqual(consulta.status_code, 200)
        self.assertEqual(len(consulta.json()["vetores"]), 1)
        documentos = self.cliente.post("/embeddings/documentos", json=DOCUMENTOS, headers=cabecalho)
        self.assertEqual(documentos.status_code, 200)
        self.assertEqual(documentos.json()["chunks"][0]["texto"], "nota privada")
        self.assertEqual(vetores.call_count, 2)

    def test_geracao_tambem_exige_header(self, _vetores):
        for rota in ("/generate-cv-pipeline", "/copiloto/turn", "/keywords"):
            resposta = self.cliente.post(rota, json={})
            self.assertEqual(resposta.status_code, 401, rota)

    def test_health_continua_publico(self, _vetores):
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
        self.assertEqual(TestClient(app).post("/embeddings/consultas", json=CONSULTA).status_code, 401)

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
