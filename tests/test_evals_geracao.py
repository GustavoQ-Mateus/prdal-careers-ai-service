import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.schemas import EstruturaCurriculo, ExperienciaEstruturada, FraseFonte
from evals import rodar
from evals.geracao import conjunto


class ConjuntoGeracaoFalsoTest(unittest.TestCase):
    def setUp(self):
        self.temporario = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporario.cleanup)
        self.saida = Path(self.temporario.name)

    def rodar(self):
        return rodar.main(["--conjunto", "geracao", "--falso", "--saida", str(self.saida)])

    def test_executor_passa_na_linha_de_base_e_registra_as_metricas(self):
        self.assertEqual(0, self.rodar())
        resumo = json.loads((self.saida / "resumo.json").read_text(encoding="utf-8"))
        metricas = resumo["metricas"]
        self.assertEqual(7, resumo["casos"])
        self.assertEqual(0, metricas["metrica_sem_fonte"])
        self.assertEqual(0, metricas["termos_proibidos_presentes"])
        self.assertEqual(0, metricas["violacoes_formato"])
        self.assertEqual(1, metricas["frases_rejeitadas"])
        self.assertEqual(1, metricas["frases_reparadas"])
        self.assertEqual(1.0, metricas["juiz_sondas"])
        self.assertEqual(1, metricas["juiz_relacao_falhas"])
        self.assertEqual("reescrita.v3", resumo["promptDaReescrita"])
        self.assertEqual("juiz_relacao.v1", resumo["juizEmProducao"])
        self.assertEqual({"ligado"}, set(resumo["juizDeProducaoPorCaso"].values()))
        self.assertEqual(0, metricas["reprovadas_juiz_producao"])
        [falha] = resumo["frasesComRelacaoNaoSustentada"]
        self.assertEqual("migracao_postgres_causalidade", falha["caso"])
        self.assertIn("reduzindo", falha["texto"])
        self.assertGreater(resumo["uso"]["custoEstimadoUsd"], 0)

        caso = json.loads((self.saida / "migracao_postgres_causalidade.json").read_text(encoding="utf-8"))
        causal = [n for n in caso["detalhes"]["notasDoJuiz"] if "reduzindo" in n["texto"]]
        self.assertEqual(1, len(causal))
        self.assertFalse(causal[0]["relacaoSustentada"])
        self.assertEqual(0, caso["metricas"]["metrica_sem_fonte"])
        self.assertEqual(3, caso["uso"]["requisicoes"])
        self.assertEqual(2, caso["detalhes"]["usoGeracao"]["requisicoes"])

        reparado = json.loads((self.saida / "backend_java_termo_ausente.json").read_text(encoding="utf-8"))
        self.assertIn("Kubernetes", reparado["detalhes"]["rejeitadas"][0]["motivo"])
        self.assertNotIn("Kubernetes", reparado["detalhes"]["markdown"])
        self.assertEqual(4, reparado["metricas"]["requisicoes"])

    def test_juiz_que_erra_a_sonda_de_causalidade_faz_o_executor_falhar(self):
        roteiros = copy.deepcopy(conjunto._roteiros())
        juiz = roteiros["migracao_postgres_causalidade"][-1]["json"]
        for nota in juiz["notas"]:
            if nota["chave"] == "sonda.1":
                nota["relacaoSustentada"] = True
        with mock.patch.object(conjunto, "_roteiros", return_value=roteiros):
            self.assertEqual(1, self.rodar())
        resumo = json.loads((self.saida / "resumo.json").read_text(encoding="utf-8"))
        self.assertIn("juiz_sondas", [r["metrica"] for r in resumo["regressoes"]])


class MetricasDeGeracaoTest(unittest.TestCase):
    def test_numero_sem_fonte_citada_e_contado(self):
        frases = [("bullet.a.1", "Reduzi o custo em 45%.", ["a"]), ("bullet.a.2", "Reduzi o custo em 30%.", ["a"])]
        achados = conjunto.metricas_sem_fonte(frases, {"a": "Reduzi o custo em 30% com cache."})
        self.assertEqual([{"chave": "bullet.a.1", "metricas": ["45%"]}], achados)

    def test_formato_aponta_secao_vazia_travessao_e_bullets_demais(self):
        caso = conjunto.carregar()[4]
        req = conjunto.requisicao(caso)
        bullets = [FraseFonte(texto=f"Item {i}", fontes=["cooperativa"]) for i in range(6)]
        estrutura = EstruturaCurriculo(experiencias=[ExperienciaEstruturada(experiencia_id="cooperativa", bullets=bullets)])
        markdown = "# Pessoa\n\n## RESUMO PROFISSIONAL\n\n## EXPERIENCIA\n- Fiz algo \u2014 bem"
        violacoes = conjunto.violacoes_de_formato(markdown, estrutura, req)
        self.assertIn("secao vazia: RESUMO PROFISSIONAL", violacoes)
        self.assertTrue(any(v.startswith("travessao") for v in violacoes))
        self.assertIn("experiencia cooperativa com 6 bullets", violacoes)

    def test_casos_cobrem_os_cenarios_pedidos_sem_dado_pessoal_real(self):
        casos = {caso["id"]: caso for caso in conjunto.carregar()}
        self.assertTrue(6 <= len(casos) <= 8)
        self.assertIn("Kubernetes", casos["backend_java_termo_ausente"]["termosProibidos"])
        self.assertFalse(casos["nota_de_apoio_isca"]["contexto"][0]["factual"])
        self.assertEqual(1, len(casos["uma_experiencia"]["perfil"]["experiencias"]))
        self.assertFalse(casos["migracao_postgres_causalidade"]["sondasJuiz"][0]["relacaoSustentada"])
        for caso in casos.values():
            for email in caso["perfil"].get("emails", []):
                self.assertTrue(email["valor"].endswith("@exemplo.dev"))


if __name__ == "__main__":
    unittest.main()
