import importlib.util
import unittest

from app import rag

PARES_RELEVANTES = (
    ("Docker", "Projeto pessoal: empacotei uma API em Docker e publiquei com docker compose."),
    ("AWS", "Migrei o processamento de arquivos para AWS Lambda e S3 no projeto da faculdade."),
    ("Kubernetes", "Operei um cluster Kubernetes com tres nos para os servicos do time."),
    ("Power BI", "Montei um painel em Power BI com indicadores de estoque para a loja."),
    ("SQL", "Otimizei consultas SQL lentas do relatorio mensal de vendas."),
    ("Salesforce", "Configurei funis e relatorios no Salesforce para a equipe comercial."),
    ("UTI", "Plantoes na UTI adulto com cuidado a pacientes criticos."),
    ("negociacao", "Conduzi a negociacao de renovacao de contratos com clientes corporativos."),
    ("Python", "Escrevi scripts em Python para automatizar a limpeza de planilhas."),
    ("React", "Construi telas em React com TypeScript para o portal do cliente."),
    ("contabilidade", "Conciliacao contabil mensal e fechamento de balancetes."),
    ("Excel", "Planilhas em Excel com tabelas dinamicas para o controle financeiro."),
)
NOTAS_SEM_RELACAO = (
    "Receita de bolo de cenoura com cobertura de chocolate.",
    "Viagem de ferias para o litoral em janeiro com a familia.",
    "Lista de compras: arroz, feijao, cafe e frutas.",
    "Resultado do jogo de futebol de domingo com os amigos.",
    "Lembrar de renovar o documento do carro em marco.",
)
KEYWORDS = tuple(termo for termo, _ in PARES_RELEVANTES) + (
    "Java", "FastAPI", "PostgreSQL", "enfermagem", "ingles", "APIs REST", "Git", "Scrum",
)


@unittest.skipUnless(importlib.util.find_spec("sentence_transformers"), "modelo de embedding nao instalado")
class LimiarComModeloRealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiar = rag.limiar_similaridade()
        cls.consultas = dict(zip(KEYWORDS, rag.vetores_de_consultas(list(KEYWORDS))))

    def _similaridade(self, termo, texto):
        vetor = rag.vetores_de_documentos([texto])[0]
        return sum(a * b for a, b in zip(self.consultas[termo], vetor))

    def test_teto_do_chunk_e_o_limite_do_modelo_e_comporta_o_chunk_inteiro(self):
        self.assertEqual(rag.limite_tokens(), rag._model().max_seq_length)
        self.assertGreaterEqual(rag.limite_tokens(), 512)
        rag.conferir_dimensao()

    def test_nota_que_trata_da_keyword_passa_do_limiar(self):
        for termo, texto in PARES_RELEVANTES:
            self.assertGreaterEqual(self._similaridade(termo, texto), self.limiar, termo)

    def test_nota_sem_relacao_fica_abaixo_do_limiar_para_toda_keyword(self):
        for texto in NOTAS_SEM_RELACAO:
            for termo in KEYWORDS:
                self.assertLess(self._similaridade(termo, texto), self.limiar, (termo, texto))


if __name__ == "__main__":
    unittest.main()
