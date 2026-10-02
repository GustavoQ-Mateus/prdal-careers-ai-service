import unittest

from app.score import SCORE_VERSAO, calcular_score
from app.schemas import Keyword

CABECALHO = "# N\n**D**\nc\n"
SECOES = "## EXPERIÊNCIA PROFISSIONAL\n## COMPETÊNCIAS\n## FORMAÇÃO ACADÊMICA\n"


def _kw(*termos: str) -> list[Keyword]:
    return [Keyword(termo=termo, peso=1) for termo in termos]


class ScoreTest(unittest.TestCase):
    def test_termos_com_pontuacao_tecnica_pontuam(self):
        md = (
            CABECALHO + "## RESUMO PROFISSIONAL\n"
            "Uso C#, .NET, Node.js, Next.js, CI/CD e C++ diariamente.\n" + SECOES
        )
        for termo in ["C#", ".NET", "Node.js", "Next.js", "CI/CD", "C++"]:
            with self.subTest(termo=termo):
                self.assertEqual(calcular_score(md, _kw(termo)).breakdown.keyword_match, 100)

    def test_composto_nao_pontua_por_componente_solto(self):
        md = CABECALHO + "## RESUMO PROFISSIONAL\nsapato de boot\n"
        self.assertEqual(calcular_score(md, _kw("Spring Boot")).breakdown.keyword_match, 0)

    def test_repeticao_nao_aumenta_o_score(self):
        kw = _kw("Python", "Docker", "AWS")
        stuffing = (
            CABECALHO + "## RESUMO PROFISSIONAL\n"
            "Python Python Python Docker Docker Docker AWS AWS AWS\n" + SECOES
        )
        honesto = CABECALHO + "## RESUMO PROFISSIONAL\nPython, Docker e AWS em produção.\n" + SECOES
        self.assertLessEqual(calcular_score(stuffing, kw).score, calcular_score(honesto, kw).score)
        uma_vez = CABECALHO + "## RESUMO PROFISSIONAL\nPython\n" + SECOES
        dez_vezes = CABECALHO + "## RESUMO PROFISSIONAL\n" + "Python " * 10 + "\n" + SECOES
        self.assertEqual(
            calcular_score(uma_vez, _kw("Python")).score,
            calcular_score(dez_vezes, _kw("Python")).score,
        )

    def test_secao_so_conta_por_heading(self):
        sem_heading = (
            CABECALHO + "Resumo profissional com experiencia em competencias e formacao academica. "
            "email: a@b.com\n"
        )
        self.assertEqual(calcular_score(sem_heading, _kw("Python")).breakdown.secoes, 0)
        com_heading = CABECALHO + "## RESUMO PROFISSIONAL\nx\n" + SECOES
        self.assertEqual(calcular_score(com_heading, _kw("Python")).breakdown.secoes, 100)

    def test_headings_em_ingles_e_espanhol_sao_reconhecidos(self):
        en = "## PROFESSIONAL SUMMARY\n## SKILLS\n## PROFESSIONAL EXPERIENCE\n## EDUCATION\n"
        es = "## RESUMEN PROFESIONAL\n## COMPETENCIAS\n## EXPERIENCIA PROFESIONAL\n## FORMACIÓN ACADÉMICA\n"
        self.assertEqual(calcular_score(en, _kw("Python")).breakdown.secoes, 100)
        self.assertEqual(calcular_score(es, _kw("Python")).breakdown.secoes, 100)

    def test_termo_em_experiencia_pesa_mais_que_so_em_competencias(self):
        em_experiencia = (
            CABECALHO + "## RESUMO PROFISSIONAL\nx\n## COMPETÊNCIAS\n- Backend: Python\n"
            "## EXPERIÊNCIA PROFISSIONAL\n**A** | Dev | 01/2024 - atual\n- Atuei com Python em producao.\n"
            "## FORMAÇÃO ACADÊMICA\nx\n"
        )
        so_competencias = (
            CABECALHO + "## RESUMO PROFISSIONAL\nx\n## COMPETÊNCIAS\n- Backend: Python\n"
            "## EXPERIÊNCIA PROFISSIONAL\n**A** | Dev | 01/2024 - atual\n- Atuei em producao.\n"
            "## FORMAÇÃO ACADÊMICA\nx\n"
        )
        forte = calcular_score(em_experiencia, _kw("Python"))
        fraco = calcular_score(so_competencias, _kw("Python"))
        self.assertGreater(forte.score, fraco.score)
        self.assertEqual(forte.breakdown.densidade, 100)
        self.assertEqual(fraco.breakdown.densidade, 0)
        self.assertEqual(fraco.breakdown.keyword_match, 100)

    def test_sinonimos_contam_uma_vez(self):
        md = CABECALHO + "## RESUMO PROFISSIONAL\nJavaScript\n" + SECOES
        resultado = calcular_score(md, _kw("JS", "JavaScript"))
        self.assertEqual(resultado.breakdown.keyword_match, 100)
        self.assertEqual(resultado.breakdown.faltando, [])

    def test_resposta_informa_versao_do_calculo(self):
        resposta = calcular_score("", _kw("Python"))
        self.assertEqual(resposta.score_versao, SCORE_VERSAO)
        self.assertEqual(resposta.model_dump(by_alias=True)["scoreVersao"], 2)
        self.assertEqual(
            set(resposta.breakdown.model_dump(by_alias=True)),
            {"keywordMatch", "densidade", "secoes", "faltando"},
        )


if __name__ == "__main__":
    unittest.main()
