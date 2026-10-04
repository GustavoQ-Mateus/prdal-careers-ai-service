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

    def test_cliente_unico_e_reutilizado_sem_retry_proprio_do_sdk(self):
        anterior = llm._cliente
        try:
            llm.definir_cliente(None)
            with patch.dict(os.environ, {"AI_MAX_RETRIES": "4", "ANTHROPIC_API_KEY": "sk-teste"}):
                primeiro = llm.cliente()
                segundo = llm.cliente()
            self.assertIs(primeiro, segundo)
            self.assertIsInstance(primeiro, anthropic.Anthropic)
            self.assertEqual(0, primeiro.max_retries)
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

    def test_erro_de_transporte_sem_retentativa_configurada_falha_na_primeira(self):
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

    def test_teto_limita_o_timeout_e_prazo_abaixo_do_piso_nao_chama(self):
        with patch.dict(os.environ, {"AI_TIMEOUT_PISO_S": "3", "AI_TIMEOUT_TETO_S": "30"}):
            with ComClienteFalso(resposta(KEYWORDS_OK), resposta(KEYWORDS_OK)) as cliente:
                with self.assertRaises(PrazoEsgotado):
                    with operacao(prazo_ms=2500):
                        _chamar()
                with operacao(prazo_ms=600000):
                    _chamar()
                _chamar()
        self.assertEqual([30.0, 30.0], [round(t, 1) for t in cliente.timeouts])
        self.assertEqual([0, 0], cliente.max_retries)

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


class RelogioVirtual:
    def __init__(self):
        self.agora = 1000.0
        self.esperas = []

    def monotonic(self):
        return self.agora

    def sleep(self, segundos):
        self.esperas.append(segundos)
        self.agora += segundos


def _estouro(relogio):
    def falhar(timeout):
        relogio.agora += timeout
        raise anthropic.APITimeoutError(httpx2.Request("POST", "https://api.anthropic.com"))

    return falhar


def _com_retry_after(segundos):
    requisicao = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resposta_http = httpx2.Response(429, request=requisicao, headers={"retry-after": str(segundos)})
    return anthropic.RateLimitError("falha", response=resposta_http, body=None)


class ClienteCronometrado:
    def __init__(self, relogio, *comportamentos):
        self.relogio = relogio
        self.comportamentos = list(comportamentos)
        self.timeouts = []
        self.max_retries = []
        self._timeout = None
        self.messages = self

    def with_options(self, timeout=None, max_retries=None, **_):
        self._timeout = timeout
        self.max_retries.append(max_retries)
        return self

    def create(self, **_):
        self.timeouts.append(self._timeout)
        comportamento = self.comportamentos.pop(0) if self.comportamentos else _estouro(self.relogio)
        if isinstance(comportamento, BaseException):
            self.relogio.agora += 0.2
            raise comportamento
        if callable(comportamento):
            return comportamento(self._timeout)
        return comportamento


class PrazoTotalDasTentativasTest(unittest.TestCase):
    def _executar(self, prazo_ms, *comportamentos, retries="2", piso="1", teto="120"):
        relogio = RelogioVirtual()
        cliente = ClienteCronometrado(relogio, *comportamentos)
        ambiente = {"AI_MAX_RETRIES": retries, "AI_TIMEOUT_PISO_S": piso, "AI_TIMEOUT_TETO_S": teto}
        with ComClienteFalso(), patch.dict(os.environ, ambiente), patch(
            "app.llm.time.monotonic", relogio.monotonic
        ), patch("app.llm.time.sleep", relogio.sleep):
            llm.definir_cliente(cliente)
            inicio = relogio.agora
            resultado, erro = None, None
            try:
                with operacao(prazo_ms=prazo_ms):
                    resultado = _chamar()
            except (LLMUnavailable, PrazoEsgotado) as exc:
                erro = exc
        return resultado, erro, relogio.agora - inicio, cliente, relogio

    def test_cliente_que_sempre_estoura_fica_dentro_do_prazo(self):
        for prazo_ms in (1500, 8000, 20000, 60000, 300000):
            _, erro, gasto, cliente, _ = self._executar(prazo_ms)
            self.assertIsInstance(erro, LLMUnavailable)
            self.assertIn("tempo esgotado", str(erro))
            self.assertLessEqual(gasto, prazo_ms / 1000)
            self.assertTrue(all(retries == 0 for retries in cliente.max_retries))

    def test_cliente_que_sempre_estoura_com_teto_curto_repete_dentro_do_prazo(self):
        with patch.dict(os.environ, {"AI_TIMEOUT_TETO_S": "10"}):
            relogio = RelogioVirtual()
            cliente = ClienteCronometrado(relogio)
            ambiente = {"AI_MAX_RETRIES": "5", "AI_TIMEOUT_PISO_S": "1"}
            with ComClienteFalso(), patch.dict(os.environ, ambiente), patch(
                "app.llm.time.monotonic", relogio.monotonic
            ), patch("app.llm.time.sleep", relogio.sleep):
                llm.definir_cliente(cliente)
                with self.assertRaises(LLMUnavailable):
                    with operacao(prazo_ms=25000):
                        _chamar()
        self.assertLessEqual(relogio.agora - 1000.0, 25)
        self.assertEqual(3, len(cliente.timeouts))
        self.assertEqual([10.0, 10.0], cliente.timeouts[:2])

    def test_falhas_rapidas_repetem_com_espera_ate_o_limite_configurado(self):
        erro_500 = _erro_status(anthropic.InternalServerError, 500)
        resultado, erro, gasto, cliente, relogio = self._executar(
            60000, erro_500, erro_500, resposta(KEYWORDS_OK)
        )
        self.assertIsNone(erro)
        self.assertEqual("Python", resultado.keywords[0].termo)
        self.assertEqual(3, len(cliente.timeouts))
        self.assertEqual([0.5, 1.0], relogio.esperas)
        self.assertLessEqual(gasto, 60)

    def test_retentativas_param_no_maximo_configurado(self):
        erro_500 = _erro_status(anthropic.InternalServerError, 500)
        _, erro, _, cliente, _ = self._executar(60000, erro_500, erro_500, erro_500, retries="1")
        self.assertIsInstance(erro, LLMUnavailable)
        self.assertEqual(2, len(cliente.timeouts))

    def test_retry_after_maior_que_o_prazo_nao_e_esperado(self):
        _, erro, gasto, cliente, relogio = self._executar(10000, _com_retry_after(60))
        self.assertIsInstance(erro, LLMUnavailable)
        self.assertIn("limite de requisicoes", str(erro))
        self.assertEqual([], relogio.esperas)
        self.assertEqual(1, len(cliente.timeouts))
        self.assertLessEqual(gasto, 10)

    def test_retry_after_que_cabe_e_respeitado_e_a_nova_tentativa_usa_o_restante(self):
        _, erro, gasto, cliente, relogio = self._executar(10000, _com_retry_after(2), resposta(KEYWORDS_OK))
        self.assertIsNone(erro)
        self.assertEqual([2.0], relogio.esperas)
        self.assertAlmostEqual(10 - 0.2 - 2.0, cliente.timeouts[1], places=3)
        self.assertLessEqual(gasto, 10)

    def test_sem_prazo_retry_after_acima_do_teto_nao_e_esperado(self):
        _, erro, gasto, cliente, relogio = self._executar(None, _com_retry_after(60), resposta(KEYWORDS_OK), teto="30")
        self.assertIsInstance(erro, LLMUnavailable)
        self.assertIn("limite de requisicoes", str(erro))
        self.assertEqual([], relogio.esperas)
        self.assertEqual(1, len(cliente.timeouts))
        self.assertLess(gasto, 1)

    def test_sem_prazo_retry_after_dentro_do_teto_e_respeitado(self):
        _, erro, _, cliente, relogio = self._executar(None, _com_retry_after(30), resposta(KEYWORDS_OK), teto="30")
        self.assertIsNone(erro)
        self.assertEqual([30.0], relogio.esperas)
        self.assertEqual([30.0, 30.0], cliente.timeouts)

    def test_nova_tentativa_sem_espaco_para_o_piso_nao_acontece(self):
        erro_500 = _erro_status(anthropic.InternalServerError, 500)
        _, erro, _, cliente, relogio = self._executar(1500, erro_500, resposta(KEYWORDS_OK), piso="1.2")
        self.assertIsInstance(erro, LLMUnavailable)
        self.assertEqual(1, len(cliente.timeouts))
        self.assertEqual([], relogio.esperas)

    def test_erro_nao_repetivel_falha_na_primeira(self):
        _, erro, _, cliente, _ = self._executar(60000, _erro_status(anthropic.AuthenticationError, 401))
        self.assertIn("credencial", str(erro))
        self.assertEqual(1, len(cliente.timeouts))

    def test_prazo_abaixo_do_piso_nao_chama(self):
        _, erro, gasto, cliente, _ = self._executar(800)
        self.assertIsInstance(erro, PrazoEsgotado)
        self.assertEqual([], cliente.timeouts)
        self.assertEqual(0, gasto)


if __name__ == "__main__":
    unittest.main()
