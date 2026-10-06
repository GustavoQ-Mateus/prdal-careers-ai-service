import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from app import llm
from app.segredos import carregar_chave


AMBIENTE = {
    "ANTHROPIC_API_KEY_SECRET_ARN": "arn:aws:secretsmanager:us-east-1:000000000000:secret:teste",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_AUTH_TOKEN": "",
    "AI_MODEL": "modelo-teste",
}
CHAVE = "credencial-falsa-do-teste"


class SegredosTest(unittest.TestCase):
    def test_arn_sem_chave_carrega_segredo_sem_logar(self):
        cliente = Mock()
        cliente.get_secret_value.return_value = {"SecretString": CHAVE}
        with patch.dict(os.environ, AMBIENTE), patch("boto3.client", return_value=cliente) as criar:
            with self.assertNoLogs("app.segredos"):
                carregar_chave()
                carregar_chave()
            self.assertEqual(CHAVE, os.environ["ANTHROPIC_API_KEY"])
            criar.assert_called_once_with("secretsmanager")
            cliente.get_secret_value.assert_called_once_with(SecretId=AMBIENTE["ANTHROPIC_API_KEY_SECRET_ARN"])

    def test_chave_definida_tem_prioridade(self):
        with patch.dict(os.environ, {**AMBIENTE, "ANTHROPIC_API_KEY": CHAVE}), patch("boto3.client") as criar:
            carregar_chave()
            self.assertEqual(CHAVE, os.environ["ANTHROPIC_API_KEY"])
            criar.assert_not_called()

    def test_sem_arn_nao_consulta(self):
        with patch.dict(os.environ, {**AMBIENTE, "ANTHROPIC_API_KEY_SECRET_ARN": ""}), patch("boto3.client") as criar:
            carregar_chave()
            criar.assert_not_called()

    def test_falha_preserva_erro_de_chave_ausente_sem_logar_excecao(self):
        cliente = Mock()
        cliente.get_secret_value.side_effect = RuntimeError(CHAVE)
        with patch.dict(os.environ, AMBIENTE), patch("boto3.client", return_value=cliente):
            with self.assertLogs("app.segredos", level="WARNING") as logs:
                carregar_chave()
            self.assertNotIn(CHAVE, "\n".join(logs.output))
            self.assertEqual("", os.environ["ANTHROPIC_API_KEY"])
            with patch.object(llm, "cliente") as provedor:
                with self.assertRaisesRegex(llm.LLMUnavailable, "^ANTHROPIC_API_KEY ausente$"):
                    llm.responder_com_tools("sistema", [], [], chamador="teste", esforco="low")
                provedor.assert_not_called()

    def test_falha_ao_criar_cliente_nao_expoe_valor(self):
        with patch.dict(os.environ, AMBIENTE), patch("boto3.client", side_effect=RuntimeError(CHAVE)):
            with self.assertLogs("app.segredos", level="WARNING") as logs:
                carregar_chave()
            self.assertNotIn(CHAVE, "\n".join(logs.output))
            self.assertFalse(llm._credencial_presente())

    def test_segredo_vazio_ou_invalido_deixa_sem_chave(self):
        for resposta in [{}, {"SecretString": " "}, {"SecretString": None}]:
            with self.subTest(resposta=resposta):
                cliente = Mock()
                cliente.get_secret_value.return_value = resposta
                with patch.dict(os.environ, AMBIENTE), patch("boto3.client", return_value=cliente):
                    with self.assertLogs("app.segredos", level="WARNING"):
                        carregar_chave()
                    self.assertFalse(llm._credencial_presente())

    def test_inicializacao_lambda_e_http_consulta_uma_vez(self):
        codigo = '''
import os
from unittest.mock import Mock, patch
cliente = Mock()
cliente.get_secret_value.return_value = {"SecretString": "credencial-falsa-do-teste"}
with patch("boto3.client", return_value=cliente) as criar:
    from app import lambda_handler, main, llm
    import app.lambda_handler
    assert os.environ["ANTHROPIC_API_KEY"] == "credencial-falsa-do-teste"
    assert llm._credencial_presente()
    criar.assert_called_once_with("secretsmanager")
    cliente.get_secret_value.assert_called_once()
'''
        resultado = subprocess.run(
            [sys.executable, "-c", codigo],
            env={**os.environ, **AMBIENTE},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, resultado.returncode, resultado.stderr)
        self.assertNotIn(CHAVE, resultado.stdout + resultado.stderr)
