import os
import unittest
from unittest.mock import patch

from app.llm import PROVEDORES_SUPORTADOS, LLMUnavailable, _sem_fence_markdown, complete_model
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


class SemFenceMarkdownTest(unittest.TestCase):
    def test_remove_fence_com_tag_json(self):
        bruto = '```json\n{"keywords": []}\n```'
        self.assertEqual('{"keywords": []}', _sem_fence_markdown(bruto))

    def test_remove_fence_sem_tag(self):
        bruto = '```\n{"markdown": "# x"}\n```'
        self.assertEqual('{"markdown": "# x"}', _sem_fence_markdown(bruto))

    def test_conteudo_sem_fence_fica_intacto(self):
        bruto = '{"keywords": []}'
        self.assertEqual('{"keywords": []}', _sem_fence_markdown(bruto))

    def test_amostra_real_claude_sonnet_4_6_via_openrouter(self):
        bruto = (
            "```json\n{\n  \"keywords\": [\n    {\"termo\": \"Java\", "
            '"peso": 1.0, "tipo": "stack"}\n  ]\n}\n```'
        )
        limpo = _sem_fence_markdown(bruto)
        self.assertTrue(limpo.startswith("{"))
        self.assertTrue(limpo.endswith("}"))


if __name__ == "__main__":
    unittest.main()
