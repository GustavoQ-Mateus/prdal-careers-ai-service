import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals import rodar
from evals.rag import conjunto


class ConjuntoRagFalsoTest(unittest.TestCase):
    def setUp(self):
        self.temporario = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporario.cleanup)
        self.saida = Path(self.temporario.name)

    def rodar(self, casos=None):
        argv = ["--conjunto", "rag", "--falso", "--saida", str(self.saida)]
        if casos is None:
            return rodar.main(argv)
        with mock.patch.object(conjunto, "carregar", return_value=casos):
            return rodar.main(argv)

    def resumo(self):
        return json.loads((self.saida / "resumo.json").read_text(encoding="utf-8"))

    def test_executor_passa_sem_chave_e_traz_a_calibracao_do_limiar(self):
        self.assertEqual(0, self.rodar())
        resumo = self.resumo()
        self.assertEqual(0, resumo["metricas"]["sem_relacao_acima_do_limiar"])
        self.assertEqual(31, resumo["metricas"]["consultas"])
        self.assertEqual(0, resumo["uso"]["requisicoes"])
        calibracao = resumo["calibracaoDoLimiar"]
        self.assertEqual(0.35, calibracao["limiarAtual"]["limiar"])
        self.assertEqual(7, len(calibracao["varredura"]))
        self.assertIn("Excel -> n-planilhas (0.2873)", calibracao["positivosAbaixoDoLimiarAtual"])
        caso = json.loads((self.saida / "corpus_variado.json").read_text(encoding="utf-8"))
        kubernetes = next(c for c in caso["detalhes"]["consultas"] if c["termo"] == "Kubernetes")
        self.assertIn("n-k8s#1", kubernetes["acimaDoLimiar"])
        self.assertNotIn("n-k8s#2", kubernetes["esperados"])

    def test_nota_sem_relacao_acima_do_limiar_faz_o_executor_falhar(self):
        casos = copy.deepcopy(conjunto.carregar())
        casos[1]["documentos"].append({"id": "s-isca", "tipo": "nota", "semRelacao": True, "texto": "Comprei um livro sobre Docker de presente."})
        self.assertEqual(1, self.rodar(casos))
        self.assertIn("sem_relacao_acima_do_limiar", [r["metrica"] for r in self.resumo()["regressoes"]])

    def test_queda_de_recall_alem_do_limiar_faz_o_executor_falhar(self):
        casos = copy.deepcopy(conjunto.carregar())
        for documento in casos[0]["documentos"]:
            if documento["id"] in ("n-docker", "n-aws", "n-git"):
                documento["texto"] = "Texto trocado sem o termo."
        self.assertEqual(1, self.rodar(casos))
        self.assertIn("recall_at_5", [r["metrica"] for r in self.resumo()["regressoes"]])


class CalibracaoTest(unittest.TestCase):
    def test_margem_negativa_quando_negativo_supera_positivo(self):
        consultas = [{
            "termo": "SQL",
            "esperados": ["a"],
            "topo": [{"id": "b", "similaridade": 0.4, "semRelacao": True}, {"id": "a", "similaridade": 0.3, "semRelacao": False}],
            "positivos": [0.3],
            "negativos": [0.4],
        }]
        calibracao = conjunto.calibracao([{"detalhes": {"limiar": 0.35, "consultas": consultas}}])
        self.assertFalse(calibracao["separavel"])
        self.assertEqual(-0.1, calibracao["margem"])
        self.assertEqual({"limiar": 0.35, "recall_at_5": 0.0, "sem_relacao_acima_do_limiar": 1, "fora_do_esperado_acima": 1}, calibracao["limiarAtual"])


if __name__ == "__main__":
    unittest.main()
