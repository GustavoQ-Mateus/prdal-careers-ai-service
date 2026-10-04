import contextlib
import io
import os
import unittest
from unittest.mock import patch

from app import llm
from scripts import sonda_claude
from tests.cliente_falso import resposta, resposta_blocos

REPARO = {
    "titulo": {"texto": "", "fontes": []},
    "resumo": [],
    "experiencias": [],
    "competencias": [],
    "reparos": [{"chave": "bullet.atual.2", "texto": "Participei da migracao do banco para PostgreSQL com o time de dados.", "fontes": ["atual"]}],
}

RESPOSTAS_POR_SCHEMA = {
    "KeywordsLlmResponse": {"keywords": [{"termo": "Python", "peso": 1, "tipo": "stack"}]},
    "MensagemLlm": {"titulo": "Contato", "texto": "Ola, tenho interesse na vaga.", "destino": "email"},
    "FormularioLlm": {"titulo": "Respostas", "respostas": [{"campo": "Por que esta vaga?", "texto": "Afinidade."}], "texto": "Afinidade."},
    "ReescritaEstruturada": {
        "titulo": {"texto": "Desenvolvedora Backend", "fontes": ["atual"]},
        "resumo": [{"texto": "Desenvolvedora backend com APIs REST em Python e Java.", "fontes": ["resumo"]}],
        "experiencias": [
            {"experienciaId": "atual", "bullets": [
                {"texto": "Desenvolvi APIs REST em Python com FastAPI para o modulo de pedidos.", "fontes": ["atual"]},
                {"texto": "Operei Kubernetes em producao.", "fontes": ["atual", "nota-planos"]},
            ]},
        ],
        "competencias": [],
        "reparos": [],
    },
}


class _Mensagens:
    def __init__(self, dono):
        self.dono = dono

    def create(self, **requisicao):
        if "tools" in requisicao:
            self.dono.titulos.append("turno")
            lida = 1500 if self.dono.titulos.count("turno") > 1 else 0
            return resposta_blocos(
                [{"type": "text", "text": "Vamos revisar a vaga."}], cache_lida=lida, cache_escrita=0 if lida else 1500
            )
        titulo = requisicao["output_config"]["format"]["schema"]["title"]
        self.dono.titulos.append(titulo)
        vezes = self.dono.titulos.count(titulo)
        lida = 1500 if vezes > 1 else 0
        conteudo = REPARO if titulo == "ReescritaEstruturada" and vezes % 2 == 0 else RESPOSTAS_POR_SCHEMA[titulo]
        return resposta(conteudo, cache_lida=lida, cache_escrita=0 if lida else 1500)


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
        resumo = next(linha for linha in linhas if linha.startswith("reescrita") and "rodada=1 resumo" in linha)
        self.assertIn("requisicoes=2", resumo)
        self.assertIn("rejeitadas=1", resumo)
        self.assertIn("descartadas=0", resumo)
        self.assertIn("cache_lida_por_chamada=[0, 1500]", resumo)
        self.assertIn("reparadas=1", resumo)
        self.assertTrue(any("fonte de apoio nao sustenta fato: nota-planos" in linha for linha in linhas))
        self.assertTrue(any(linha.startswith("# Pessoa Exemplo") for linha in linhas))


if __name__ == "__main__":
    unittest.main()
