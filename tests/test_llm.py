import os
import unittest
from unittest.mock import patch

import anthropic
import httpx2
from fastapi.testclient import TestClient

from app import llm
from app.llm import (
    LLMUnavailable,
    ModeloAusente,
    PrazoEsgotado,
    ValidacaoSemantica,
    complete_model,
    exigir_modelo_no_boot,
    operacao,
)
from app.main import app
from app.schemas import KeywordsLlmResponse, ReescritaLlm
from tests.cliente_falso import ComClienteFalso, resposta

KEYWORDS_OK = {"keywords": [{"termo": "Python", "peso": 1.0, "tipo": "stack"}]}


def _erro_status(classe, status):
    requisicao = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return classe("falha", response=httpx2.Response(status, request=requisicao), body=None)


def _chamar(schema=KeywordsLlmResponse, **kwargs):
    return complete_model("sistema fixo", "entrada variavel", schema, chamador="teste", esforco="low", **kwargs)


class ConfiguracaoDoModeloTest(unittest.TestCase):
    def test_sem_ai_model_degrada_sem_chamar(self):
        with ComClienteFalso() as cliente, patch.dict(os.environ, {"AI_MODEL": ""}):
            with self.assertRaises(LLMUnavailable) as ctx:
                _chamar()
        self.assertIn("AI_MODEL", str(ctx.exception))
        self.assertEqual([], cliente.requisicoes)

    def test_sem_chave_degrada_sem_chamar(self):
        with ComClienteFalso() as cliente, patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            with self.assertRaises(LLMUnavailable) as ctx:
                _chamar()
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))
        self.assertEqual([], cliente.requisicoes)

    def test_fora_de_desenvolvimento_sem_modelo_nao_sobe(self):
        with patch.dict(os.environ, {"PRDAL_AMBIENTE": "producao", "AI_MODEL": ""}):
            with self.assertRaises(ModeloAusente):
                exigir_modelo_no_boot()
        with patch.dict(os.environ, {"PRDAL_AMBIENTE": "producao", "AI_MODEL": "claude-sonnet-5"}):
            exigir_modelo_no_boot()
        with patch.dict(os.environ, {"PRDAL_AMBIENTE": "desenvolvimento", "AI_MODEL": ""}):
            exigir_modelo_no_boot()

    def test_codigo_nao_tem_modelo_padrao_nem_outro_provedor(self):
        fonte = open(llm.__file__, encoding="utf-8").read()
        for proibido in ("claude-", "openai", "groq", "openrouter", "temperature", "_sem_fence"):
            self.assertNotIn(proibido, fonte)

    def test_cliente_unico_e_reutilizado_com_retry_configurado(self):
        anterior = llm._cliente
        try:
            llm.definir_cliente(None)
            with patch.dict(os.environ, {"AI_MAX_RETRIES": "4", "ANTHROPIC_API_KEY": "sk-teste"}):
                primeiro = llm.cliente()
                segundo = llm.cliente()
            self.assertIs(primeiro, segundo)
            self.assertIsInstance(primeiro, anthropic.Anthropic)
            self.assertEqual(4, primeiro.max_retries)
        finally:
            llm.definir_cliente(anterior)


class RequisicaoNativaTest(unittest.TestCase):
    def test_saida_estruturada_nativa_com_esforco_e_sem_amostragem(self):
        with ComClienteFalso(resposta(KEYWORDS_OK), modelo="modelo-configurado") as cliente:
            resultado = _chamar()
        self.assertEqual("Python", resultado.keywords[0].termo)
        requisicao = cliente.requisicoes[0]
        self.assertEqual("modelo-configurado", requisicao["model"])
        self.assertEqual("low", requisicao["output_config"]["effort"])
        formato = requisicao["output_config"]["format"]
        self.assertEqual("json_schema", formato["type"])
        self.assertFalse(formato["schema"]["additionalProperties"])
        for proibido in ("temperature", "top_p", "top_k", "thinking", "output_format", "tools", "tool_choice"):
            self.assertNotIn(proibido, requisicao)

    def test_esforco_desconhecido_e_erro_de_programacao(self):
        with ComClienteFalso() as cliente:
            with self.assertRaises(ValueError):
                complete_model("s", "u", ReescritaLlm, chamador="teste", esforco="altissimo")
        self.assertEqual([], cliente.requisicoes)


class StopReasonTest(unittest.TestCase):
    def test_recusa_vira_indisponibilidade_com_motivo_e_sem_reparo(self):
        with ComClienteFalso(resposta("", stop_reason="refusal", categoria="cyber")) as cliente:
            with self.assertRaises(LLMUnavailable) as ctx:
                _chamar()
        self.assertIn("recusou", str(ctx.exception))
        self.assertIn("cyber", str(ctx.exception))
        self.assertEqual(1, len(cliente.requisicoes))

    def test_limite_de_tokens_nunca_vira_parse_parcial(self):
        with ComClienteFalso(resposta('{"keywords": [', stop_reason="max_tokens")) as cliente:
            with self.assertRaises(LLMUnavailable) as ctx:
                _chamar()
        self.assertIn("limite de tokens", str(ctx.exception))
        self.assertEqual(1, len(cliente.requisicoes))


class ReparoTest(unittest.TestCase):
    def test_schema_invalido_gera_uma_nova_chamada_com_o_erro(self):
        invalido = {"keywords": [{"termo": "Python", "peso": "alto", "tipo": "stack"}]}
        with ComClienteFalso(resposta(invalido), resposta(KEYWORDS_OK)) as cliente:
            resultado = _chamar()
        self.assertEqual(1.0, resultado.keywords[0].peso)
        self.assertEqual(2, len(cliente.requisicoes))
        primeira, segunda = cliente.requisicoes
        self.assertEqual(primeira["system"], segunda["system"])
        self.assertEqual(primeira["messages"][0], segunda["messages"][0])
        self.assertEqual("assistant", segunda["messages"][1]["role"])
        reparo = segunda["messages"][2]["content"][0]["text"]
        self.assertIn("nao passou na validacao", reparo)
        self.assertIn("peso", reparo)

    def test_regra_do_chamador_tambem_gera_reparo(self):
        def exigir_duas(res):
            if len(res.keywords) < 2:
                raise ValidacaoSemantica("preciso de pelo menos duas palavras-chave")

        duas = {"keywords": KEYWORDS_OK["keywords"] * 2}
        with ComClienteFalso(resposta(KEYWORDS_OK), resposta(duas)) as cliente:
            resultado = _chamar(validar=exigir_duas)
        self.assertEqual(2, len(resultado.keywords))
        self.assertIn("pelo menos duas", cliente.requisicoes[1]["messages"][2]["content"][0]["text"])

    def test_reparo_e_unico(self):
        with ComClienteFalso(resposta({"x": 1}), resposta({"y": 2})) as cliente:
            with self.assertRaises(LLMUnavailable) as ctx:
                _chamar()
        self.assertIn("apos reparo", str(ctx.exception))
        self.assertEqual(2, len(cliente.requisicoes))

    def test_erro_de_transporte_fica_com_o_retry_do_sdk(self):
        with ComClienteFalso(_erro_status(anthropic.RateLimitError, 429)) as cliente:
            with self.assertRaises(LLMUnavailable) as ctx:
                _chamar()
        self.assertIn("limite de requisicoes", str(ctx.exception))
        self.assertEqual(1, len(cliente.requisicoes))

    def test_erros_tipados_viram_motivo_legivel(self):
        casos = [
            (_erro_status(anthropic.InternalServerError, 500), "status=500"),
            (_erro_status(anthropic.AuthenticationError, 401), "credencial"),
            (anthropic.APITimeoutError(httpx2.Request("POST", "https://api.anthropic.com")), "tempo esgotado"),
            (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), "conexao"),
        ]
        for erro, trecho in casos:
            with ComClienteFalso(erro):
                with self.assertRaises(LLMUnavailable) as ctx:
                    _chamar()
            self.assertIn(trecho, str(ctx.exception))


class PrazoTest(unittest.TestCase):
    def test_prazo_da_api_vira_timeout_da_chamada(self):
        with patch.dict(os.environ, {"AI_TIMEOUT_PISO_S": "1", "AI_TIMEOUT_TETO_S": "60"}):
            with ComClienteFalso(resposta(KEYWORDS_OK)) as cliente, operacao(prazo_ms=20000):
                _chamar()
        self.assertLessEqual(cliente.timeouts[0], 20)
        self.assertGreater(cliente.timeouts[0], 19)

    def test_piso_e_teto_limitam_o_timeout(self):
        with patch.dict(os.environ, {"AI_TIMEOUT_PISO_S": "3", "AI_TIMEOUT_TETO_S": "30"}):
            with ComClienteFalso(resposta(KEYWORDS_OK), resposta(KEYWORDS_OK), resposta(KEYWORDS_OK)) as cliente:
                with operacao(prazo_ms=500):
                    _chamar()
                with operacao(prazo_ms=600000):
                    _chamar()
                _chamar()
        self.assertEqual([3.0, 30.0, 30.0], [round(t, 1) for t in cliente.timeouts])

    def test_prazo_vencido_nao_chama_o_modelo(self):
        with ComClienteFalso() as cliente:
            with self.assertRaises(PrazoEsgotado):
                with operacao(prazo_ms=0):
                    _chamar()
            with patch("app.llm.time.monotonic", side_effect=[100.0, 200.0]):
                with self.assertRaises(PrazoEsgotado):
                    with operacao(prazo_ms=1000):
                        _chamar()
        self.assertEqual([], cliente.requisicoes)

    def test_rota_responde_504_sem_chamar_quando_o_prazo_acabou(self):
        with ComClienteFalso() as cliente:
            resposta_http = TestClient(app).post(
                "/keywords", json={"descricao": "Python"}, headers={"X-Prdal-Prazo-Ms": "0"}
            )
        self.assertEqual(504, resposta_http.status_code)
        self.assertEqual([], cliente.requisicoes)

    def test_rota_repassa_o_prazo_ao_cliente(self):
        with patch.dict(os.environ, {"AI_TIMEOUT_PISO_S": "1", "AI_TIMEOUT_TETO_S": "120"}):
            with ComClienteFalso(resposta(KEYWORDS_OK)) as cliente:
                resposta_http = TestClient(app).post(
                    "/keywords", json={"descricao": "Python"}, headers={"X-Prdal-Prazo-Ms": "8000"}
                )
        self.assertEqual(200, resposta_http.status_code)
        self.assertLessEqual(cliente.timeouts[0], 8)


if __name__ == "__main__":
    unittest.main()
