import json
import logging
import os
import unittest
from unittest.mock import patch

import anthropic
import httpx2
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import telemetria
from app.llm import LLMUnavailable, complete_model, operacao
from app.main import app
from app.schemas import KeywordsLlmResponse
from tests.cliente_falso import ComClienteFalso, resposta, resposta_blocos

KEYWORDS_OK = {"keywords": [{"termo": "Python", "peso": 1.0, "tipo": "stack"}]}
SISTEMA = "instrucao fixa secreta"
ENTRADA = "Maria Pereira, maria@exemplo.dev, vaga de Python"


def _chamar(**kwargs):
    return complete_model(SISTEMA, ENTRADA, KeywordsLlmResponse, chamador="keywords", esforco="low", **kwargs)


class SpanTest(unittest.TestCase):
    def setUp(self):
        self.exportador = InMemorySpanExporter()
        provedor = TracerProvider()
        provedor.add_span_processor(SimpleSpanProcessor(self.exportador))
        telemetria.configurar(provedor)

    def tearDown(self):
        telemetria.configurar(None)

    def test_um_span_por_chamada_com_convencoes_genai(self):
        with ComClienteFalso(
            resposta(KEYWORDS_OK, entrada=40, saida=12, cache_lida=1500, cache_escrita=0, modelo="claude-sonnet-5"),
            modelo="claude-sonnet-5",
        ):
            with operacao(operacao_id="conversa:abc"):
                _chamar(prompt_version="keywords.v1")
        spans = self.exportador.get_finished_spans()
        self.assertEqual(1, len(spans))
        span = spans[0]
        atributos = dict(span.attributes)
        self.assertEqual("chat claude-sonnet-5", span.name)
        self.assertEqual("claude-sonnet-5", atributos["gen_ai.request.model"])
        self.assertEqual("claude-sonnet-5", atributos["gen_ai.response.model"])
        self.assertEqual(1540, atributos["gen_ai.usage.input_tokens"])
        self.assertEqual(12, atributos["gen_ai.usage.output_tokens"])
        self.assertEqual(1500, atributos["gen_ai.usage.cache_read.input_tokens"])
        self.assertEqual(0, atributos["gen_ai.usage.cache_creation.input_tokens"])
        self.assertEqual("ok", atributos["prdal.resultado"])
        self.assertEqual("keywords.v1", atributos["prdal.prompt_version"])
        self.assertEqual("keywords", atributos["prdal.chamador"])
        self.assertEqual("conversa:abc", atributos["prdal.operacao_id"])
        self.assertIn("prdal.latencia_ms", atributos)
        self.assertEqual(("end_turn",), tuple(atributos["gen_ai.response.finish_reasons"]))

    def test_span_nao_leva_prompt_nem_dado_pessoal(self):
        with ComClienteFalso(resposta(KEYWORDS_OK)):
            _chamar()
        serializado = json.dumps({k: str(v) for k, v in self.exportador.get_finished_spans()[0].attributes.items()})
        for proibido in (SISTEMA, "Maria", "maria@exemplo.dev", "Python"):
            self.assertNotIn(proibido, serializado)

    def test_prompt_version_padrao_e_estavel_por_texto(self):
        with ComClienteFalso(resposta(KEYWORDS_OK), resposta(KEYWORDS_OK)):
            _chamar()
            _chamar()
        primeira, segunda = self.exportador.get_finished_spans()
        self.assertEqual(primeira.attributes["prdal.prompt_version"], segunda.attributes["prdal.prompt_version"])
        self.assertTrue(primeira.attributes["prdal.prompt_version"].startswith("keywords."))

    def test_resultados_recusa_limite_e_erro(self):
        requisicao = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        erro = anthropic.InternalServerError("falha", response=httpx2.Response(500, request=requisicao), body=None)
        casos = [
            (resposta("", stop_reason="refusal"), "recusa"),
            (resposta("{", stop_reason="max_tokens"), "limite"),
            (erro, "erro"),
        ]
        for programada, esperado in casos:
            with ComClienteFalso(programada):
                with self.assertRaises(LLMUnavailable):
                    _chamar()
            self.assertEqual(esperado, self.exportador.get_finished_spans()[-1].attributes["prdal.resultado"])
        self.assertEqual("InternalServerError", self.exportador.get_finished_spans()[-1].attributes["error.type"])

    def test_reparo_gera_segundo_span_marcado(self):
        with ComClienteFalso(resposta({"keywords": "x"}), resposta(KEYWORDS_OK)):
            _chamar()
        tentativas = [span.attributes["prdal.tentativa"] for span in self.exportador.get_finished_spans()]
        self.assertEqual([1, 2], tentativas)


class LogSemExportadorTest(unittest.TestCase):
    def test_sem_endpoint_uma_linha_de_log_estruturada_por_chamada(self):
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}):
            telemetria.configurar()
        self.assertFalse(telemetria.exportando())
        with self.assertLogs("prdal.llm", level=logging.INFO) as registro:
            with ComClienteFalso(resposta(KEYWORDS_OK), resposta(KEYWORDS_OK)):
                _chamar()
                _chamar()
        self.assertEqual(2, len(registro.records))
        linha = json.loads(registro.records[0].getMessage())
        self.assertEqual("chamada_llm", linha["evento"])
        self.assertEqual("ok", linha["prdal.resultado"])
        self.assertNotIn(SISTEMA, registro.records[0].getMessage())
        self.assertNotIn("Maria", registro.records[0].getMessage())

    def test_com_endpoint_liga_o_exportador_otlp(self):
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318"}):
            telemetria.configurar()
        try:
            self.assertTrue(telemetria.exportando())
        finally:
            telemetria.encerrar()
            telemetria.configurar(None)


class UsoDevolvidoTest(unittest.TestCase):
    def test_rota_devolve_uso_agregado_e_modelo(self):
        with ComClienteFalso(
            resposta({"keywords": []}, entrada=10, saida=5, cache_escrita=900),
            resposta(KEYWORDS_OK, entrada=12, saida=7, cache_lida=900),
            modelo="claude-sonnet-5",
        ):
            corpo = TestClient(app).post("/keywords", json={"descricao": "Python"}).json()
        self.assertEqual(
            {"entrada": 22, "saida": 12, "cacheLida": 900, "cacheEscrita": 900, "chamadas": 2},
            corpo["uso"],
        )
        self.assertEqual("claude-teste", corpo["modelo"])

    def test_indisponibilidade_tambem_devolve_o_uso_gasto(self):
        vazio = resposta_blocos([{"type": "text", "text": ""}], entrada=30)
        with ComClienteFalso(vazio):
            resposta_http = TestClient(app).post(
                "/copiloto/turn", json={"mensagens": [{"role": "user", "content": [{"type": "text", "text": "oi"}]}]}
            )
        self.assertEqual(503, resposta_http.status_code)
        self.assertEqual(30, resposta_http.json()["uso"]["entrada"])
        self.assertEqual(1, resposta_http.json()["uso"]["chamadas"])

    def test_geracao_devolve_modelo_e_prompt_version(self):
        corpo_req = {
            "perfilMestre": {"nome": "Pessoa", "experiencias": [], "skills": ["Python"]},
            "vaga": {"titulo": "Backend", "empresa": "Empresa", "descricao": "Python"},
            "keywords": [{"termo": "Python", "peso": 1}],
        }
        estrutura = {
            "titulo": {"texto": "Backend", "fontes": ["skills"]},
            "resumo": [{"texto": "Experiencia com Python.", "fontes": ["skills"]}],
            "experiencias": [],
            "competencias": [],
            "reparos": [],
        }
        with ComClienteFalso(*[resposta(estrutura) for _ in range(4)]):
            corpo = TestClient(app).post("/generate-cv-pipeline", json=corpo_req).json()
        self.assertEqual("claude-teste", corpo["modelo"])
        self.assertEqual("reescrita.v2", corpo["promptVersion"])
        self.assertEqual("Experiencia com Python.", corpo["estrutura"]["resumo"][0]["texto"])
        self.assertGreaterEqual(corpo["uso"]["chamadas"], 1)

    def test_sem_chamada_o_modelo_fica_vazio(self):
        with patch.dict(os.environ, {"AI_MODEL": ""}):
            corpo = TestClient(app).post("/keywords", json={"descricao": "Python"}).json()
        self.assertIsNone(corpo["modelo"])
        self.assertEqual(0, corpo["uso"]["chamadas"])


if __name__ == "__main__":
    unittest.main()
