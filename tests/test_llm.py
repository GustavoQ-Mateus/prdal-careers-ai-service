import os
import unittest
from unittest.mock import patch

from app.llm import PROVEDORES_SUPORTADOS, LLMUnavailable, complete_model
from app.schemas import GenerateCvResponse


class ProvedorAnthropicTest(unittest.TestCase):
    def test_anthropic_esta_na_lista_de_provedores_suportados(self):
        self.assertIn("anthropic", PROVEDORES_SUPORTADOS)

    @patch.dict(os.environ, {"AI_PROVIDER": "anthropic"}, clear=False)
    def test_sem_chave_cai_em_erro_sem_quebrar(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)

        with self.assertRaises(LLMUnavailable) as ctx:
            complete_model("sistema", "usuario", GenerateCvResponse)

        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    @patch.dict(
        os.environ, {"AI_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-test"}
    )
    @patch("app.llm._completar_anthropic")
    def test_com_chave_delega_para_o_sdk_anthropic(self, mock_completar):
        mock_completar.return_value = GenerateCvResponse(markdown="# x")

        resultado = complete_model("sistema", "usuario", GenerateCvResponse)

        mock_completar.assert_called_once()
        self.assertEqual("# x", resultado.markdown)

    @patch.dict(os.environ, {"AI_PROVIDER": "invalido"}, clear=False)
    def test_provedor_desconhecido_continua_indisponivel(self):
        with self.assertRaises(LLMUnavailable):
            complete_model("sistema", "usuario", GenerateCvResponse)


if __name__ == "__main__":
    unittest.main()
