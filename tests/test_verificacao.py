import unittest

from app.fontes import fontes_da_geracao
from app.schemas import FonteContexto, FraseFonte
from app.verificacao import (
    motivo_da_competencia,
    motivo_da_rejeicao,
    nomes_proprios,
    permissao_do_bullet,
    termos_reconhecidos,
)
from tests.perfis import req_dados, req_dev


def _motivo(req, texto, fontes, experiencia_id=None):
    permissao = permissao_do_bullet(experiencia_id) if experiencia_id else None
    return motivo_da_rejeicao(
        FraseFonte(texto=texto, fontes=list(fontes)),
        fontes_da_geracao(req),
        termos_reconhecidos(req),
        permissao,
    )


class FontePorFraseTest(unittest.TestCase):
    def test_java_atribuido_a_experiencia_errada_e_rejeitado(self):
        req = req_dev()
        motivo = _motivo(req, "Desenvolvi APIs RESTful em Java (Spring Boot) para inspecoes.", ["rota"], "rota")
        self.assertIsNotNone(motivo)
        self.assertIn("Java", motivo)
        self.assertIn("Spring Boot", motivo)

    def test_bullet_que_cita_outra_experiencia_e_rejeitado(self):
        req = req_dev()
        motivo = _motivo(req, "Atuei com Java (Spring Boot) e FastAPI.", ["rota", "erp"], "rota")
        self.assertEqual("bullet cita outra experiencia: erp", motivo)

    def test_bullet_so_cita_a_propria_experiencia_ou_nota_factual(self):
        req = req_dev()
        self.assertIn("so pode citar", _motivo(req, "Atuei com Python.", ["skills"], "rota"))
        self.assertIsNone(_motivo(req, "Atuei com Python (FastAPI) e PostgreSQL em producao.", ["rota"], "rota"))

    def test_nota_sem_marca_nao_autoriza_kubernetes(self):
        nota = FonteContexto(id="n1", tipo="nota", factual=False, titulo="Planos", texto="Quero estudar Kubernetes.")
        req = req_dev(contexto=[nota])
        self.assertEqual("fonte de apoio nao sustenta fato: n1", _motivo(req, "Experiencia com Kubernetes.", ["n1"]))
        self.assertIn("Kubernetes", _motivo(req, "Experiencia com Kubernetes.", ["resumo"]))
        self.assertIn("Kubernetes", _motivo(req, "Operei Kubernetes em producao.", ["rota"], "rota"))

    def test_nota_marcada_como_historico_autoriza_e_pode_ser_citada_no_bullet(self):
        nota = FonteContexto(id="n2", tipo="nota", factual=True, titulo="Projeto", texto="Operei Kubernetes em producao no projeto de inspecoes.")
        req = req_dev(contexto=[nota])
        self.assertIsNone(_motivo(req, "Operei Kubernetes em producao.", ["n2"]))
        self.assertIsNone(_motivo(req, "Operei Kubernetes em producao nas inspecoes.", ["rota", "n2"], "rota"))

    def test_candidatura_nunca_e_fato_mesmo_marcada(self):
        candidatura = FonteContexto(id="c1", tipo="candidatura", factual=True, texto="Recrutador pediu Kubernetes.")
        req = req_dev(contexto=[candidatura])
        self.assertEqual("fonte de apoio nao sustenta fato: c1", _motivo(req, "Kubernetes.", ["c1"]))

    def test_metrica_sem_fonte_e_rejeitada(self):
        req = req_dev()
        self.assertIn("60%", _motivo(req, "Reduzi o tempo de resposta em 60% com PostgreSQL.", ["rota"], "rota"))
        self.assertIsNone(_motivo(req, "Reduzi o tempo de resposta das consultas em 40% com PostgreSQL.", ["rota"], "rota"))
        self.assertIn("40%", _motivo(req, "Reduzi o tempo de resposta em 40%.", ["erp"], "erp"))

    def test_numero_com_unidade_e_moeda_precisam_da_fonte(self):
        req = req_dados()
        self.assertIsNone(_motivo(req, "Consolidei dados de 120 lojas com SQL.", ["varejo"], "varejo"))
        self.assertIn("200 lojas", _motivo(req, "Consolidei dados de 200 lojas com SQL.", ["varejo"], "varejo"))
        self.assertIn("R$ 2", _motivo(req, "Economizei R$ 2 milhoes com SQL.", ["varejo"], "varejo"))

    def test_fonte_inexistente_e_frase_sem_fonte(self):
        req = req_dev()
        self.assertEqual("fonte inexistente: x9", _motivo(req, "Atuei com Python.", ["x9"]))
        self.assertEqual("frase sem fonte citada", _motivo(req, "Atuei com Python.", []))

    def test_vocabulario_interno_da_redacao_e_rejeitado(self):
        req = req_dev()
        self.assertIn("resultado:", _motivo(req, "Atuei com Python. Resultado: entrega.", ["rota"], "rota"))

    def test_sinonimo_do_dicionario_conta_como_o_mesmo_termo(self):
        nota = FonteContexto(id="n3", tipo="nota", factual=True, texto="Mantive clusters k8s do time.")
        req = req_dev(contexto=[nota])
        self.assertIsNone(_motivo(req, "Mantive clusters Kubernetes do time.", ["n3"]))

    def test_termo_do_dicionario_fora_da_vaga_e_do_perfil_tambem_e_verificado(self):
        req = req_dev()
        self.assertIn("TypeScript", _motivo(req, "Atuei com TypeScript.", ["rota"], "rota"))

    def test_competencia_precisa_estar_escrita_na_fonte(self):
        fontes = fontes_da_geracao(req_dev())
        self.assertIsNone(motivo_da_competencia("Spring Boot", "erp", fontes))
        self.assertIn("ausente", motivo_da_competencia("Spring Boot", "rota", fontes))
        self.assertIn("inexistente", motivo_da_competencia("Java", "nada", fontes))


class OutraProfissaoTest(unittest.TestCase):
    def test_analista_de_dados_verifica_sem_catalogo_de_desenvolvedor(self):
        req = req_dados()
        self.assertIsNone(_motivo(req, "Construi paineis de vendas em Power BI para a diretoria.", ["varejo"], "varejo"))
        self.assertIn("Power BI", _motivo(req, "Mantive paineis em Power BI.", ["banco"], "banco"))

    def test_nome_proprio_fora_das_listas_tambem_precisa_da_fonte(self):
        req = req_dados()
        self.assertIn("Tableau", _motivo(req, "Construi paineis em Tableau.", ["varejo"], "varejo"))
        self.assertIn("Python", _motivo(req, "Automatizei relatorios em Python.", ["varejo"], "varejo"))
        self.assertIn("Fantasma", _motivo(req, "Atendi a conta da Loja Fantasma com SQL.", ["varejo"], "varejo"))

    def test_maiuscula_no_inicio_da_frase_nao_e_nome_proprio(self):
        self.assertEqual(["Power", "BI"], nomes_proprios("Construi paineis. Depois usei Power BI."))
        self.assertEqual(["S3", "C#"], nomes_proprios("migrei arquivos para S3 e C# em 2023."))
        self.assertEqual([], nomes_proprios("Atuei com dados. Escrevi consultas."))


if __name__ == "__main__":
    unittest.main()
