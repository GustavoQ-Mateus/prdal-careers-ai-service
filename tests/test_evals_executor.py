import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from evals import rodar
from evals.falso import cliente_falso
from evals.nucleo import Gravador, Uso, comparar, gravando, medir
from app.llm import complete_model
from app.schemas import ResumoLlm

REGRAS = {
    "acerto": {"direcao": "maior", "limiar": 0.1, "valor": 0.9},
    "violacoes": {"direcao": "menor", "absoluto": 0, "valor": 0},
    "passos": {"direcao": "menor", "limiar": 1, "valor": None},
}


def conjunto_de_teste(acerto, violacoes=0, falha=None):
    modulo = types.ModuleType("evals_conjunto_teste")
    modulo.EXIGE_CREDENCIAL = True

    def carregar():
        return [{"id": "um"}, {"id": "dois"}]

    def rodar_caso(caso, falso):
        if falha == caso["id"]:
            raise RuntimeError("caso quebrado")
        return {"id": caso["id"], "metricas": {"acerto": acerto, "violacoes": violacoes}, "uso": Uso(requisicoes=1, entrada=1000, saida=100)}

    def resumir(resultados):
        return {
            "acerto": sum(r["metricas"]["acerto"] for r in resultados) / len(resultados),
            "violacoes": sum(r["metricas"]["violacoes"] for r in resultados),
            "passos": 2,
        }

    modulo.carregar = carregar
    modulo.rodar_caso = rodar_caso
    modulo.resumir = resumir
    return modulo


class ComparacaoTest(unittest.TestCase):
    def test_queda_dentro_do_limiar_passa_e_alem_dele_falha(self):
        self.assertEqual([], comparar({"acerto": 0.81, "violacoes": 0, "passos": 9}, REGRAS))
        regressoes = comparar({"acerto": 0.79, "violacoes": 0, "passos": 9}, REGRAS)
        self.assertEqual(["acerto"], [r.metrica for r in regressoes])

    def test_limite_absoluto_vale_mesmo_sem_linha_de_base(self):
        regras = {"violacoes": {"direcao": "menor", "absoluto": 0, "valor": None}}
        self.assertEqual(["violacoes"], [r.metrica for r in comparar({"violacoes": 1}, regras)])

    def test_metrica_sem_valor_de_referencia_nao_bloqueia(self):
        self.assertEqual([], comparar({"passos": 50}, {"passos": REGRAS["passos"]}))

    def test_metrica_ausente_com_referencia_e_regressao(self):
        self.assertEqual(["acerto"], [r.metrica for r in comparar({}, {"acerto": REGRAS["acerto"]})])


class UsoTest(unittest.TestCase):
    def test_custo_usa_precos_por_milhao_configuraveis(self):
        uso = Uso(requisicoes=1, entrada=1_000_000, saida=100_000, cache_lida=1_000_000, cache_escrita=0)
        self.assertEqual(2.0 + 1.0 + 0.2, uso.custo_usd())
        os.environ["EVAL_PRECO_ENTRADA_MTOK"] = "3"
        try:
            self.assertEqual(3.0 + 1.0 + 0.2, uso.custo_usd())
        finally:
            os.environ.pop("EVAL_PRECO_ENTRADA_MTOK")

    def test_medir_soma_requisicoes_e_tokens_e_o_gravador_guarda_o_corpo(self):
        uso = Uso()
        with cliente_falso([{"json": {"resumo": "ok"}, "uso": {"entrada": 50, "saida": 7, "cache_lida": 30}}]):
            with gravando() as gravador, medir(uso):
                complete_model("sistema", "usuario", ResumoLlm, chamador="teste", esforco="low")
        self.assertIsInstance(gravador, Gravador)
        self.assertEqual(1, len(gravador.requisicoes))
        self.assertEqual("sistema", gravador.requisicoes[0].corpo["system"][0]["text"])
        self.assertEqual({"requisicoes": 1, "entrada": 50, "saida": 7, "cacheLida": 30, "cacheEscrita": 0}, {k: v for k, v in uso.como_dict().items() if k != "custoEstimadoUsd"})


class ExecutorTest(unittest.TestCase):
    def setUp(self):
        self.temporario = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporario.cleanup)
        self.dir = Path(self.temporario.name)
        self.base = self.dir / "base.json"
        self.base.write_text(json.dumps({"teste": {"metricas": REGRAS}}), encoding="utf-8")
        self.anterior = dict(rodar.CONJUNTOS)
        rodar.CONJUNTOS["teste"] = "evals_conjunto_teste"
        self.addCleanup(self._restaurar)

    def _restaurar(self):
        rodar.CONJUNTOS.clear()
        rodar.CONJUNTOS.update(self.anterior)
        sys.modules.pop("evals_conjunto_teste", None)

    def rodar(self, modulo, *extra):
        sys.modules["evals_conjunto_teste"] = modulo
        saida = self.dir / "saida"
        return rodar.main(["--conjunto", "teste", "--falso", "--saida", str(saida), "--linha-de-base", str(self.base), *extra]), saida

    def test_grava_relatorio_por_caso_e_resumo_e_sai_com_zero(self):
        codigo, saida = self.rodar(conjunto_de_teste(0.85))
        self.assertEqual(0, codigo)
        caso = json.loads((saida / "um.json").read_text(encoding="utf-8"))
        self.assertEqual(1000, caso["uso"]["entrada"])
        self.assertGreater(caso["uso"]["custoEstimadoUsd"], 0)
        resumo = json.loads((saida / "resumo.json").read_text(encoding="utf-8"))
        self.assertEqual(2, resumo["casos"])
        self.assertEqual(2, resumo["uso"]["requisicoes"])
        self.assertEqual([], resumo["regressoes"])

    def test_regressao_forcada_sai_com_erro(self):
        codigo, saida = self.rodar(conjunto_de_teste(0.5))
        self.assertEqual(1, codigo)
        resumo = json.loads((saida / "resumo.json").read_text(encoding="utf-8"))
        self.assertEqual("acerto", resumo["regressoes"][0]["metrica"])

    def test_limite_absoluto_violado_sai_com_erro(self):
        codigo, _ = self.rodar(conjunto_de_teste(0.95, violacoes=1))
        self.assertEqual(1, codigo)

    def test_caso_que_quebra_aparece_no_resumo_e_falha_a_execucao(self):
        codigo, saida = self.rodar(conjunto_de_teste(0.95, falha="dois"))
        self.assertEqual(1, codigo)
        resumo = json.loads((saida / "resumo.json").read_text(encoding="utf-8"))
        self.assertEqual("dois", resumo["erros"][0]["id"])

    def test_gravar_linha_de_base_atualiza_os_valores_e_mantem_as_regras(self):
        codigo, _ = self.rodar(conjunto_de_teste(0.7), "--gravar-linha-de-base")
        self.assertEqual(0, codigo)
        base = json.loads(self.base.read_text(encoding="utf-8"))["teste"]
        self.assertEqual(0.7, base["metricas"]["acerto"]["valor"])
        self.assertEqual(0.1, base["metricas"]["acerto"]["limiar"])
        self.assertEqual(2, base["metricas"]["passos"]["valor"])
        self.assertEqual("falso", base["modelo"])

    def test_sem_credencial_ao_vivo_sai_com_dois_sem_rodar(self):
        sys.modules["evals_conjunto_teste"] = conjunto_de_teste(1.0)
        guardado = {chave: os.environ.pop(chave, None) for chave in ("ANTHROPIC_API_KEY", "AI_MODEL")}
        try:
            codigo = rodar.main(["--conjunto", "teste", "--saida", str(self.dir / "x"), "--linha-de-base", str(self.base)])
        finally:
            for chave, valor in guardado.items():
                if valor is not None:
                    os.environ[chave] = valor
        self.assertEqual(2, codigo)
        self.assertFalse((self.dir / "x").exists())

    def test_linhas_de_base_versionadas_tem_os_tres_conjuntos(self):
        for caminho in (rodar.LINHA_DE_BASE, rodar.LINHA_DE_BASE_FALSO):
            base = json.loads(caminho.read_text(encoding="utf-8"))
            self.assertEqual({"geracao", "agente", "rag"}, set(base))
            for conjunto in base.values():
                for regra in conjunto["metricas"].values():
                    self.assertIn(regra["direcao"], ("maior", "menor"))


if __name__ == "__main__":
    unittest.main()
