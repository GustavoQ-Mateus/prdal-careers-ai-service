import unittest
from dataclasses import fields
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import degradacao as deg
from app.generate import generate_cv_pipeline
from app.llm import LLMUnavailable
from app.main import app
from app.schemas import GenerateCvRequest

PROVEDORES = ("groq", "openrouter", "anthropic", "claude", "openai", "sonnet", "gpt")


def _frases() -> list[deg.Degradacao]:
    return [valor for valor in vars(deg).values() if isinstance(valor, deg.Degradacao)]


def _req() -> GenerateCvRequest:
    return GenerateCvRequest.model_validate(
        {
            "perfilMestre": {
                "nome": "Pessoa",
                "skills": ["Python"],
                "experiencias": [
                    {"empresa": "A", "cargo": "Dev", "periodo": "01/2024 - atual", "descricao": "- Atuei com Python."}
                ],
            },
            "vaga": {"titulo": "Dev Python", "descricao": "Python"},
            "keywords": [{"termo": "Python", "peso": 1}],
        }
    )


class DegradacaoTest(unittest.TestCase):
    def test_toda_degradacao_tem_codigo_e_frase(self):
        self.assertTrue(_frases())
        codigos = [d.codigo for d in _frases()]
        self.assertEqual(len(codigos), len(set(codigos)))
        for d in _frases():
            with self.subTest(codigo=d.codigo):
                self.assertEqual({f.name for f in fields(d)}, {"codigo", "frase"})
                self.assertTrue(d.codigo and d.frase)

    def test_nenhuma_frase_nomeia_provedor(self):
        for d in _frases():
            with self.subTest(codigo=d.codigo):
                self.assertFalse(any(p in d.frase.lower() for p in PROVEDORES))

    @patch("app.generate.complete_model", side_effect=LLMUnavailable("timeout do provedor"))
    def test_indisponibilidade_grava_frase_e_loga_codigo(self, _complete):
        with self.assertLogs("app.degradacao", level="WARNING") as logs:
            resultado = generate_cv_pipeline(_req())
        self.assertEqual(resultado.degradacao, deg.REESCRITA_INDISPONIVEL.frase)
        self.assertIn(f"codigo={deg.REESCRITA_INDISPONIVEL.codigo}", logs.output[0])
        self.assertIn("timeout do provedor", logs.output[0])

    @patch("app.main.extract_keywords", side_effect=LLMUnavailable("429"))
    def test_keywords_indisponiveis_devolvem_frase_de_produto(self, _extract):
        resposta = TestClient(app).post("/keywords", json={"descricao": "vaga"})
        self.assertEqual(resposta.json()["status"], "PENDENTE")
        self.assertEqual(resposta.json()["degradacao"], deg.KEYWORDS_INDISPONIVEIS.frase)


if __name__ == "__main__":
    unittest.main()
