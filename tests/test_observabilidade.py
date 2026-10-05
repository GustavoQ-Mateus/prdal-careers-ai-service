import json
import logging
import signal
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import observabilidade
from app.main import app
from app.observabilidade import FormatadorJson, MENSAGEM_DESLIGAMENTO, avisar_desligamento, desligando

CORPO_SCORE = {"markdown": "# Pessoa\n## RESUMO\nPython", "vaga": {"titulo": "Dev", "descricao": "Python", "keywords": [{"termo": "Python", "peso": 1}]}}
TURNO = {"mensagens": [{"role": "user", "content": [{"type": "text", "text": "oi"}]}]}


class Captura(logging.Handler):
    def __init__(self):
        super().__init__()
        self.linhas: list[dict] = []
        self.setFormatter(FormatadorJson())

    def emit(self, registro):
        self.linhas.append(json.loads(self.format(registro)))


class RequestIdTest(unittest.TestCase):
    def setUp(self):
        self.captura = Captura()
        logging.getLogger("prdal.http").addHandler(self.captura)
        self.addCleanup(logging.getLogger("prdal.http").removeHandler, self.captura)
        nivel = logging.getLogger("prdal.http").level
        logging.getLogger("prdal.http").setLevel(logging.INFO)
        self.addCleanup(logging.getLogger("prdal.http").setLevel, nivel)

    def test_request_id_recebido_volta_no_cabecalho_e_vai_no_log_json(self):
        resposta = TestClient(app).post("/score", json=CORPO_SCORE, headers={"X-Request-Id": "req-api-77"})
        self.assertEqual(200, resposta.status_code)
        self.assertEqual("req-api-77", resposta.headers["x-request-id"])
        acesso = next(linha for linha in self.captura.linhas if linha["mensagem"] == "requisicao")
        self.assertEqual("req-api-77", acesso["requestId"])
        self.assertEqual("ai-service", acesso["servico"])
        self.assertEqual("POST", acesso["metodo"])
        self.assertEqual("/score", acesso["rota"])
        self.assertEqual(200, acesso["status"])
        self.assertIsInstance(acesso["duracaoMs"], int)
        self.assertTrue({"horario", "nivel", "contexto"} <= set(acesso))

    def test_sem_request_id_valido_um_novo_e_gerado(self):
        resposta = TestClient(app).post("/score", json=CORPO_SCORE, headers={"X-Request-Id": "com espaco"})
        self.assertRegex(resposta.headers["x-request-id"], r"^[0-9a-f-]{36}$")

    def test_log_dentro_da_requisicao_leva_o_request_id(self):
        captura = Captura()
        logging.getLogger("app.main").addHandler(captura)
        self.addCleanup(logging.getLogger("app.main").removeHandler, captura)
        from app.score import calcular_score

        def calcular_com_log(markdown, keywords):
            logging.getLogger("app.main").warning("calculando")
            return calcular_score(markdown, keywords)

        with patch("app.main.calcular_score", side_effect=calcular_com_log):
            TestClient(app).post("/score", json=CORPO_SCORE, headers={"X-Request-Id": "req-dentro"})
        linha = next(linha for linha in captura.linhas if linha["mensagem"] == "calculando")
        self.assertEqual("req-dentro", linha["requestId"])

    def test_formatador_inclui_stack_de_excecao(self):
        captura = Captura()
        registro = logging.getLogger("teste")
        registro.addHandler(captura)
        self.addCleanup(registro.removeHandler, captura)
        try:
            raise ValueError("quebrou")
        except ValueError:
            registro.exception("falhou")
        self.assertIn("ValueError: quebrou", captura.linhas[0]["stack"])
        self.assertEqual("error", captura.linhas[0]["nivel"])


class DesligamentoTest(unittest.TestCase):
    def tearDown(self):
        desligando.clear()

    def test_stream_aberto_termina_com_erro_recuperavel_quando_o_servico_desliga(self):
        def turno_lento(_req, emitir, _ao_uso=None):
            emitir("Comecei ")
            desligando.set()
            threading.Event().wait(0.5)
            emitir("depois do desligamento")

        with patch("app.main.planejar_turno_em_stream", side_effect=turno_lento):
            resposta = TestClient(app).post("/copiloto/turn/stream", json=TURNO)
        linhas = [json.loads(linha) for linha in resposta.text.splitlines() if linha.strip()]
        self.assertEqual(["delta", "erro"], [linha["tipo"] for linha in linhas])
        self.assertEqual(MENSAGEM_DESLIGAMENTO, linhas[-1]["detail"])
        self.assertTrue(linhas[-1]["interrompido"])

    def test_sinal_marca_o_desligamento_e_chama_o_tratador_do_uvicorn(self):
        chamados = []
        anteriores = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        self.addCleanup(lambda: [signal.signal(s, h) for s, h in anteriores.items()])
        signal.signal(signal.SIGTERM, lambda numero, _quadro: chamados.append(numero))
        avisar_desligamento()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertTrue(desligando.is_set())
        self.assertEqual([signal.SIGTERM], chamados)

    def test_entrada_da_prazo_ao_desligamento_gracioso(self):
        import os
        import runpy

        with patch.dict(os.environ, {"AWS_LAMBDA_RUNTIME_API": ""}), patch("os.execvp") as executar:
            runpy.run_module("app.entrada", run_name="__main__")
        argumentos = executar.call_args.args[1]
        self.assertEqual("uvicorn", executar.call_args.args[0])
        self.assertEqual("25", argumentos[argumentos.index("--timeout-graceful-shutdown") + 1])


if __name__ == "__main__":
    unittest.main()
