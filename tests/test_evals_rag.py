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

    def test_executor_passa_sem_chave_e_so_o_trecho_que_casa_a_keyword_entra(self):
        self.assertEqual(0, self.rodar())
        resumo = self.resumo()
        self.assertEqual(0, resumo["metricas"]["sem_relacao_no_contexto"])
        self.assertEqual(83, resumo["metricas"]["consultas"])
        self.assertEqual(0.6562, resumo["metricas"]["recall_at_5_antigas"])
        self.assertEqual(0, resumo["uso"]["requisicoes"])
        self.assertIn("Excel -> n-planilhas", " ".join(resumo["esperadosForaDoContexto"]))
        caso = json.loads((self.saida / "corpus_variado.json").read_text(encoding="utf-8"))
        kubernetes = next(c for c in caso["detalhes"]["consultas"] if c["termo"] == "Kubernetes")
        self.assertIn("n-k8s#1", kubernetes["contexto"])
        self.assertNotIn("n-k8s#2", kubernetes["contexto"])

    def test_nota_sem_relacao_com_a_keyword_no_contexto_faz_o_executor_falhar(self):
        casos = copy.deepcopy(conjunto.carregar())
        casos[1]["documentos"].append({"id": "s-isca", "tipo": "nota", "semRelacao": True, "texto": "Comprei um livro sobre Docker de presente."})
        self.assertEqual(1, self.rodar(casos))
        self.assertIn("sem_relacao_no_contexto", [r["metrica"] for r in self.resumo()["regressoes"]])

    def test_qualquer_queda_de_recall_faz_o_executor_falhar(self):
        casos = copy.deepcopy(conjunto.carregar())
        for documento in casos[0]["documentos"]:
            if documento["id"] in ("n-docker", "n-aws", "n-git"):
                documento["texto"] = "Texto trocado sem o termo."
        self.assertEqual(1, self.rodar(casos))
        self.assertIn("recall_at_5", [r["metrica"] for r in self.resumo()["regressoes"]])


class PortaoTest(unittest.TestCase):
    def test_similaridade_so_ordena_e_o_casamento_de_termos_decide(self):
        topo = [
            {"id": "parecido", "texto": "Montei planilhas com tabelas dinamicas."},
            {"id": "literal", "texto": "Usei Excel no fechamento."},
            {"id": "sinonimo", "texto": "Operei o cluster k8s do time."},
            {"id": "vazio", "texto": ""},
        ]
        self.assertEqual(["literal"], conjunto.no_contexto("Excel", topo))
        self.assertEqual(["sinonimo"], conjunto.no_contexto("Kubernetes", topo))


if __name__ == "__main__":
    unittest.main()
