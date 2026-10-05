import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import app

TOKEN = "token-de-servico-de-teste-com-mais-de-32-bytes"
AMBIENTE = {"PRDAL_AMBIENTE": "producao", "SERVICE_TOKEN": TOKEN, "AI_MODEL": "claude-teste", "ANTHROPIC_API_KEY": "sk-teste"}


def _modelo(carregado: bool):
    return SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=1 if carregado else 0))


def _chroma_ok(*_args, **_kwargs):
    return httpx.Response(200, request=httpx.Request("GET", "http://chroma/api/v1/heartbeat"))


def _chroma_fora(*_args, **_kwargs):
    raise httpx.ConnectError("conexao recusada")


@patch.dict(os.environ, AMBIENTE)
class ProntidaoTest(unittest.TestCase):
    def setUp(self):
        self.cliente = TestClient(app)

    def _dependencias(self, resposta):
        for d in resposta.json()["dependencias"]:
            self.assertEqual(["nome", "obrigatoria", "estado"], list(d))
        return {d["nome"]: d for d in resposta.json()["dependencias"]}

    def _motivos(self, chamar):
        with self.assertLogs("prdal.prontidao", "WARNING") as capturados:
            resposta = chamar()
        return resposta, {r.dependencia: r for r in capturados.records}

    def test_tudo_no_ar_responde_pronto_sem_token_de_servico(self):
        with patch("app.prontidao.httpx.get", side_effect=_chroma_ok), patch("app.prontidao.rag._model", _modelo(True)):
            resposta = self.cliente.get("/ready")
        self.assertEqual(200, resposta.status_code)
        self.assertEqual("pronto", resposta.json()["status"])
        deps = self._dependencias(resposta)
        self.assertEqual(["prompts", "claude", "chroma", "embeddings"], list(deps))
        self.assertTrue(all(d["estado"] == "ok" for d in deps.values()))
        self.assertTrue(deps["claude"]["obrigatoria"])
        self.assertFalse(deps["chroma"]["obrigatoria"])

    def test_chroma_fora_e_opcional_e_aparece_indisponivel(self):
        with patch("app.prontidao.httpx.get", side_effect=_chroma_fora), patch("app.prontidao.rag._model", _modelo(False)):
            resposta, motivos = self._motivos(lambda: self.cliente.get("/ready", headers={"X-Request-Id": "req-ready"}))
        self.assertEqual(200, resposta.status_code)
        deps = self._dependencias(resposta)
        self.assertEqual("indisponivel", deps["chroma"]["estado"])
        self.assertIn("conexao recusada", motivos["chroma"].detalhe)
        self.assertNotIn("conexao recusada", resposta.text)
        from app.observabilidade import FormatadorJson
        import json

        linha = json.loads(FormatadorJson().format(motivos["chroma"]))
        self.assertEqual("req-ready", linha["requestId"])
        self.assertEqual("chroma", linha["dependencia"])
        self.assertEqual("indisponivel", deps["embeddings"]["estado"])

    def test_sem_credencial_do_claude_responde_503(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}), patch(
            "app.prontidao.httpx.get", side_effect=_chroma_ok
        ), patch("app.prontidao.rag._model", _modelo(True)):
            resposta, motivos = self._motivos(lambda: self.cliente.get("/ready"))
        self.assertEqual(503, resposta.status_code)
        self.assertEqual("indisponivel", resposta.json()["status"])
        self.assertEqual("indisponivel", self._dependencias(resposta)["claude"]["estado"])
        self.assertEqual("ANTHROPIC_API_KEY ausente", motivos["claude"].detalhe)

    def test_health_so_diz_que_o_processo_esta_vivo(self):
        with patch("app.prontidao.httpx.get", side_effect=_chroma_fora):
            self.assertEqual({"service": "ai-service", "status": "ok"}, self.cliente.get("/health").json())


if __name__ == "__main__":
    unittest.main()
