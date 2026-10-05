import unittest
from unittest.mock import patch

from app.lambda_handler import handler
from app.passos import executar
from tests.cliente_falso import ComClienteFalso, resposta
from tests.perfis import frase, req_dev, reescrita


RASCUNHO = reescrita(
    frase("Desenvolvedora Back-End", "rota"),
    [frase("Desenvolvedora back-end com APIs REST em Python e Java.", "resumo")],
    [("erp", [frase("Atuei em modulos ERP com Java (Spring Boot) sobre MySQL.", "erp")])],
)


class PassosTest(unittest.TestCase):
    def test_lambda_despacha_cada_passo(self):
        for passo in ("rascunho", "verificar", "reparar", "montar", "keywords"):
            with self.subTest(passo=passo), patch("app.lambda_handler.executar", return_value={"ok": True}) as executar_passo:
                self.assertEqual({"ok": True}, handler({"passo": passo, "payload": {"x": 1}, "prazoMs": 5000}, None))
                executar_passo.assert_called_once_with(passo, {"x": 1}, 5000, None)

    def test_sequencia_sem_estado_e_mesma_entrada_para_lambda(self):
        dados = req_dev().model_dump(by_alias=True)
        with patch.dict("os.environ", {"AI_JUIZ_RELACAO": "0"}), ComClienteFalso(resposta(RASCUNHO)) as cliente:
            rascunho = executar("rascunho", dados, 30000)
            verificacao = handler({
                "passo": "verificar", "payload": {**dados, "rascunho": rascunho["rascunho"], "chamadasRestantes": 5},
                "prazoMs": 30000,
            }, None)
            montagem = handler({"passo": "montar", "payload": {**dados, "estado": verificacao["estado"]}}, None)
        self.assertEqual(1, len(cliente.requisicoes))
        self.assertEqual([], verificacao["rejeitadas"])
        self.assertIn("Java", montagem["markdown"])
        self.assertEqual("reescrita.v4", montagem["promptVersion"])

    def test_lambda_recusa_passo_desconhecido(self):
        with self.assertRaisesRegex(ValueError, "passo desconhecido"):
            handler({"passo": "desconhecido", "payload": {}}, None)
