import os
import unittest
from unittest.mock import patch

from app import rag
from app.schemas import Documento


def _contar_palavras(texto: str) -> int:
    return len(texto.split())


class _ColecaoFalsa:
    def __init__(self, respostas):
        self.respostas = respostas
        self.consultas = []
        self.adicionados = []

    def query(self, **kwargs):
        self.consultas.append(kwargs)
        n = len(kwargs["query_embeddings"])
        docs, metas, dists = [], [], []
        for i in range(n):
            linhas = self.respostas[i] if i < len(self.respostas) else []
            docs.append([texto for texto, _, _ in linhas])
            metas.append([meta for _, meta, _ in linhas])
            dists.append([dist for _, _, dist in linhas])
        return {"documents": docs, "metadatas": metas, "distances": dists}

    def delete(self, **_):
        pass

    def add(self, **kwargs):
        self.adicionados.append(kwargs)


def _meta(fonte_id, tipo, factual):
    return {"fonteId": fonte_id, "origemId": fonte_id, "tipo": tipo, "factual": factual, "titulo": fonte_id, "origem": "perfil"}


def _distancia(similaridade):
    return 2.0 * (1.0 - similaridade)


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
        experiencia = Documento(usuario_id="u", origem="perfil", origem_id="exp-a", tipo="experiencia", factual=True, texto="x")
        nota = Documento(usuario_id="u", origem="nota", origem_id="n1", tipo="nota", texto="x")
        self.assertEqual(rag.id_da_fonte(experiencia, 0, 1), "exp-a")
        self.assertEqual(rag.id_da_fonte(nota, 0, 1), "n1")
        self.assertEqual(rag.id_da_fonte(nota, 1, 3), "n1#2")

    def test_indexacao_grava_tipo_factual_e_id_da_fonte(self):
        colecao = _ColecaoFalsa([])
        doc = Documento(usuario_id="u", origem="perfil", origem_id="exp-a", tipo="experiencia", factual=True, titulo="T", texto="Analista de dados")
        with patch.object(rag, "_collection", return_value=colecao), patch.object(rag, "_embeddings", side_effect=lambda t: [[0.0]] * len(t)):
            rag.indexar([doc])
        meta = colecao.adicionados[0]["metadatas"][0]
        self.assertEqual((meta["fonteId"], meta["tipo"], meta["factual"]), ("exp-a", "experiencia", True))


class ConsultaTest(unittest.TestCase):
    def _consultar(self, respostas, consultas, limiar=None):
        colecao = _ColecaoFalsa(respostas)
        ambiente = {"RAG_LIMIAR_SIMILARIDADE": str(limiar)} if limiar is not None else {}
        with patch.object(rag, "_collection", return_value=colecao), patch.object(
            rag, "_embeddings", side_effect=lambda textos: [[0.0]] * len(textos)
        ), patch.dict(os.environ, ambiente):
            return rag.consultar("u", consultas, 5), colecao

    def test_uma_consulta_por_keyword_com_uniao_e_deduplicacao(self):
        respostas = [
            [("SQL e Power BI", _meta("exp-a", "experiencia", True), _distancia(0.7))],
            [("SQL e Power BI", _meta("exp-a", "experiencia", True), _distancia(0.8)), ("Python", _meta("skills", "skills", True), _distancia(0.6))],
        ]
        resultado, colecao = self._consultar(respostas, ["SQL", "Power BI", "sql", " "])
        self.assertEqual(len(colecao.consultas[0]["query_embeddings"]), 2)
        self.assertEqual([c.id for c in resultado.chunks], ["exp-a", "skills"])
        self.assertEqual(resultado.chunks[0].similaridade, 0.8)
        self.assertEqual(resultado.chunks[0].tipo, "experiencia")
        self.assertTrue(resultado.chunks[0].factual)

    def test_consultas_respeitam_o_teto(self):
        _, colecao = self._consultar([], [f"termo {i}" for i in range(40)])
        self.assertEqual(len(colecao.consultas[0]["query_embeddings"]), rag.TETO_CONSULTAS)

    def test_chunk_abaixo_do_limiar_nao_entra(self):
        respostas = [[
            ("quero estudar Kubernetes", _meta("n1", "nota", False), _distancia(0.2)),
            ("Modelei paineis em Power BI", _meta("exp-a", "experiencia", True), _distancia(0.5)),
        ]]
        resultado, _ = self._consultar(respostas, ["Power BI"], limiar=0.35)
        self.assertEqual([c.id for c in resultado.chunks], ["exp-a"])

    def test_limiar_vem_do_ambiente_e_valor_invalido_usa_o_padrao(self):
        with patch.dict(os.environ, {"RAG_LIMIAR_SIMILARIDADE": "0.6"}):
            self.assertEqual(rag.limiar_similaridade(), 0.6)
        with patch.dict(os.environ, {"RAG_LIMIAR_SIMILARIDADE": "muito"}):
            self.assertEqual(rag.limiar_similaridade(), rag.LIMIAR_PADRAO)
        with patch.dict(os.environ, {"RAG_LIMIAR_SIMILARIDADE": "7"}):
            self.assertEqual(rag.limiar_similaridade(), rag.LIMIAR_PADRAO)

    def test_sem_consulta_nao_vai_ao_chroma(self):
        resultado, colecao = self._consultar([], ["", "  "])
        self.assertEqual(resultado.chunks, [])
        self.assertEqual(colecao.consultas, [])


if __name__ == "__main__":
    unittest.main()
