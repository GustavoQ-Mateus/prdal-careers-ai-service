import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import rag
from app.main import app
from app.schemas import DocumentoParaEmbedding, TrechoCandidato


def _contar_palavras(texto: str) -> int:
    return len(texto.split())


class DivisaoTest(unittest.TestCase):
    def test_unidade_do_perfil_vira_um_chunk_inteiro(self):
        texto = "Cargo: Analista\n\nEmpresa: A\n\n" + " ".join(["palavra"] * 500)
        for tipo in ("experiencia", "resumo", "skills", "formacao", "certificacao", "idiomas"):
            self.assertEqual(rag.dividir(texto, tipo, limite=10, contar=_contar_palavras), [texto.strip()])

    def test_nota_quebra_por_paragrafo(self):
        partes = rag.dividir("Primeiro paragrafo.\n\nSegundo paragrafo.", "nota", limite=50, contar=_contar_palavras)
        self.assertEqual(partes, ["Primeiro paragrafo.", "Segundo paragrafo."])

    def test_paragrafo_acima_do_teto_quebra_por_frase_e_nenhum_pedaco_passa_do_teto(self):
        paragrafo = " ".join(f"Frase numero {i} com cinco palavras." for i in range(20))
        partes = rag.dividir(paragrafo, "nota", limite=12, contar=_contar_palavras)
        self.assertGreater(len(partes), 1)
        self.assertTrue(all(_contar_palavras(p) <= 12 for p in partes))
        self.assertEqual(" ".join(partes), paragrafo)

    def test_frase_unica_acima_do_teto_quebra_por_palavra(self):
        frase = " ".join(f"p{i}" for i in range(30))
        partes = rag.dividir(frase, "candidatura", limite=8, contar=_contar_palavras)
        self.assertTrue(all(_contar_palavras(p) <= 8 for p in partes))
        self.assertEqual(" ".join(partes), frase)

    def test_teto_padrao_vem_do_modelo(self):
        class Modelo:
            max_seq_length = 7

        with patch.object(rag, "_model", return_value=Modelo()):
            self.assertEqual(rag.limite_tokens(), 7)

    def test_id_da_experiencia_e_o_id_do_perfil_e_nota_numera_os_pedacos(self):
        self.assertEqual(rag.id_da_fonte("exp-a", "experiencia", 0, 1), "exp-a")
        self.assertEqual(rag.id_da_fonte("n1", "nota", 0, 1), "n1")
        self.assertEqual(rag.id_da_fonte("n1", "nota", 1, 3), "n1#2")


class ChunksTest(unittest.TestCase):
    def test_chunks_levam_documento_indice_fonte_e_vetor_sem_guardar_nada(self):
        documentos = [
            DocumentoParaEmbedding(id="d1", origem_id="exp-a", tipo="experiencia", texto="Analista de dados"),
            DocumentoParaEmbedding(id="d2", origem_id="n1", tipo="nota", texto="Primeiro.\n\nSegundo."),
            DocumentoParaEmbedding(id="d3", origem_id="n2", tipo="nota", texto="   "),
        ]
        embutidos = []

        def embutir(textos):
            embutidos.append(textos)
            return [[float(i)] for i in range(len(textos))]

        chunks = rag.chunks_dos_documentos(documentos, limite=50, contar=_contar_palavras, embutir=embutir)
        self.assertEqual(embutidos, [["Analista de dados", "Primeiro.", "Segundo."]])
        self.assertEqual(
            [(c.documento_id, c.indice, c.fonte_id, c.vetor) for c in chunks],
            [("d1", 0, "exp-a", [0.0]), ("d2", 0, "n1#1", [1.0]), ("d2", 1, "n1#2", [2.0])],
        )

    def test_documento_e_consulta_recebem_os_prefixos_do_modelo(self):
        recebidos = []

        def codificar(textos, normalize_embeddings):
            self.assertTrue(normalize_embeddings)
            recebidos.append(textos)
            return SimpleNamespace(tolist=lambda: [[1.0]] * len(textos))

        with patch.object(rag, "_model", return_value=SimpleNamespace(encode=codificar)), patch.dict(os.environ, {"EMBED_MODEL": "intfloat/multilingual-e5-small"}):
            rag.vetores_de_documentos(["Docker em producao"])
            rag.vetores_de_consultas([" Docker "])
            self.assertEqual(rag.vetores_de_consultas([]), [])
        self.assertEqual(recebidos, [["passage: Docker em producao"], ["query: experiência com Docker"]])


class ConfiguracaoTest(unittest.TestCase):
    def test_modelo_padrao_e_e5_small_e_desconhecido_e_recusado(self):
        with patch.dict(os.environ, {"EMBED_MODEL": ""}):
            self.assertEqual(rag.configuracao().nome, "intfloat/multilingual-e5-small")
        with patch.dict(os.environ, {"EMBED_MODEL": "modelo-sem-calibracao"}), self.assertRaises(rag.ModeloDesconhecido):
            rag.configuracao()

    def test_dimensao_diferente_da_do_banco_derruba_o_boot(self):
        modelo = SimpleNamespace(get_embedding_dimension=lambda: 768)
        with patch.object(rag, "_model", return_value=modelo), patch.dict(os.environ, {"EMBED_MODEL": "intfloat/multilingual-e5-small", "EMBED_DIMENSAO": "384"}):
            with self.assertRaises(rag.DimensaoIncompativel):
                rag.conferir_dimensao()
        modelo = SimpleNamespace(get_embedding_dimension=lambda: 384)
        with patch.object(rag, "_model", return_value=modelo), patch.dict(os.environ, {"EMBED_MODEL": "intfloat/multilingual-e5-small", "EMBED_DIMENSAO": "768"}):
            with self.assertRaises(rag.DimensaoIncompativel):
                rag.conferir_dimensao()
        with patch.object(rag, "_model", return_value=modelo), patch.dict(os.environ, {"EMBED_DIMENSAO": "384"}):
            rag.conferir_dimensao()


class FiltroTest(unittest.TestCase):
    def setUp(self):
        self.cliente = TestClient(app)

    def trechos(self, *pares):
        return [TrechoCandidato(id=id_, texto=texto) for id_, texto in pares]

    def test_so_passa_o_trecho_que_casa_a_keyword_inclusive_por_sinonimo(self):
        trechos = self.trechos(
            ("literal", "Operei um cluster Kubernetes com Helm."),
            ("sinonimo", "Subi os servicos no k8s do time."),
            ("parecido", "Orquestrei conteineres com rollouts graduais."),
        )
        self.assertEqual(rag.trechos_que_casam("Kubernetes", trechos), ["literal", "sinonimo"])

    def test_termo_ausente_ou_trecho_vazio_nao_passa(self):
        trechos = self.trechos(("vazio", ""), ("outro", "Montei planilhas com tabelas dinamicas."))
        self.assertEqual(rag.trechos_que_casam("Excel", trechos), [])
        self.assertEqual(rag.trechos_que_casam("", self.trechos(("a", "Excel"))), [])

    def test_rota_filtra_cada_consulta_sem_carregar_o_modelo(self):
        with patch.object(rag, "_model", side_effect=AssertionError("nao deve carregar o modelo")):
            resposta = self.cliente.post("/rag/filtrar", json={"consultas": [
                {"consulta": "Excel", "trechos": [{"id": "n1", "texto": "Planilhas em Excel."}, {"id": "n2", "texto": "Livro de receitas."}]},
                {"consulta": "Node.js", "trechos": [{"id": "n3", "texto": "APIs em NodeJS."}, {"id": "n4", "texto": ""}]},
            ]})
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.json(), {"consultas": [{"consulta": "Excel", "aceitos": ["n1"]}, {"consulta": "Node.js", "aceitos": ["n3"]}]})


class RotasTest(unittest.TestCase):
    def setUp(self):
        self.cliente = TestClient(app)

    def test_rotas_devolvem_modelo_e_dimensao_sem_limiar(self):
        with patch.object(rag, "_vetores", side_effect=lambda textos: [[0.5] * 384 for _ in textos]), patch.dict(os.environ, {"EMBED_MODEL": ""}):
            docs = self.cliente.post("/embeddings/documentos", json={"documentos": [{"id": "d1", "origemId": "exp-a", "tipo": "experiencia", "texto": "SQL"}]})
            consultas = self.cliente.post("/embeddings/consultas", json={"consultas": ["SQL", "Power BI"]})
        self.assertEqual(docs.status_code, 200)
        corpo = docs.json()
        self.assertEqual((corpo["modelo"], corpo["dimensao"]), ("intfloat/multilingual-e5-small", 384))
        self.assertEqual(corpo["chunks"][0]["documentoId"], "d1")
        self.assertEqual(corpo["chunks"][0]["fonteId"], "exp-a")
        self.assertEqual(len(consultas.json()["vetores"]), 2)
        self.assertNotIn("limiar", consultas.json())

    def test_modelo_fora_responde_503(self):
        with patch.object(rag, "_vetores", side_effect=OSError("sem modelo")):
            resposta = self.cliente.post("/embeddings/consultas", json={"consultas": ["SQL"]})
        self.assertEqual(resposta.status_code, 503)
        self.assertNotIn("sem modelo", resposta.text)

    def test_rotas_antigas_de_contexto_nao_existem_mais(self):
        for rota in ("/context/ingest", "/context/replace", "/context/query"):
            self.assertEqual(self.cliente.post(rota, json={}).status_code, 404, rota)


if __name__ == "__main__":
    unittest.main()
