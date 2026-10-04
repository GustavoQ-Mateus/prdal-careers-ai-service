import contextlib
import io
import os
import unittest
from unittest.mock import patch

from app import llm
from scripts import sonda_claude
from tests.cliente_falso import resposta

RESPOSTAS_POR_SCHEMA = {
    "KeywordsLlmResponse": {"keywords": [{"termo": "Python", "peso": 1, "tipo": "stack"}]},
    "TurnoLlm": {"tipo": "texto", "texto": "Vamos revisar a vaga.", "tool": None, "argsJson": None},
    "MensagemLlm": {"titulo": "Contato", "texto": "Ola, tenho interesse na vaga.", "destino": "email"},
    "FormularioLlm": {"titulo": "Respostas", "respostas": [{"campo": "Por que esta vaga?", "texto": "Afinidade."}], "texto": "Afinidade."},
    "ReescritaEstruturada": {
        "titulo": {"texto": "Desenvolvedora Backend", "fontes": ["experiencia-1"]},
        "resumo": [{"texto": "Desenvolvedora backend com APIs REST em Python.", "fontes": ["resumo"]}],
        "experiencias": [],
        "competencias": [],
        "reparos": [],
    },
}


class _Mensagens:
    def __init__(self, dono):
        self.dono = dono

    def create(self, **requisicao):
        titulo = requisicao["output_config"]["format"]["schema"]["title"]
        self.dono.titulos.append(titulo)
        lida = 1500 if self.dono.titulos.count(titulo) > 1 else 0
        return resposta(RESPOSTAS_POR_SCHEMA[titulo], cache_lida=lida, cache_escrita=0 if lida else 1500)


class ClientePorSchema:
    def __init__(self):
        self.titulos = []
        self.messages = _Mensagens(self)

    def with_options(self, **_):
        return self


class SondaTest(unittest.TestCase):
    def test_sem_chave_sai_com_codigo_diferente_de_zero(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "AI_MODEL": "claude-sonnet-5"}):
            erro = io.StringIO()
            with contextlib.redirect_stderr(erro):
                self.assertNotEqual(0, sonda_claude.main())
        self.assertIn("ANTHROPIC_API_KEY", erro.getvalue())

    def test_sem_modelo_sai_com_codigo_diferente_de_zero(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-teste", "AI_MODEL": ""}):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertNotEqual(0, sonda_claude.main())

    def test_imprime_uma_linha_por_chamada_com_tokens_cache_e_validacao(self):
        anterior = llm._cliente
        cliente = ClientePorSchema()
        llm.definir_cliente(cliente)
        saida = io.StringIO()
        try:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-teste", "AI_MODEL": "claude-sonnet-5"}):
                with contextlib.redirect_stdout(saida):
                    sonda_claude.main()
        finally:
            llm.definir_cliente(anterior)
        linhas = saida.getvalue().splitlines()
        for chamador in ("keywords", "copiloto_turno", "redigir_mensagem", "redigir_formulario", "reescrita"):
            do_chamador = [linha for linha in linhas if linha.startswith(chamador)]
            self.assertTrue(any("rodada=1" in linha for linha in do_chamador), chamador)
            self.assertTrue(any("rodada=2" in linha for linha in do_chamador), chamador)
        primeira = next(linha for linha in linhas if linha.startswith("keywords") and "rodada=1" in linha)
        segunda = next(linha for linha in linhas if linha.startswith("keywords") and "rodada=2" in linha)
        for campo in ("modelo=", "entrada=", "saida=", "cache_escrita=", "cache_lida=", "latencia_ms=", "validou=sim"):
            self.assertIn(campo, primeira)
        self.assertIn("cache_escrita=1500", primeira)
        self.assertIn("cache_lida=1500", segunda)


if __name__ == "__main__":
    unittest.main()
