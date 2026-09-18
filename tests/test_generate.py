import unittest
from unittest.mock import patch

from app.generate import (
    _erros_contrato,
    _erros_coerencia,
    _erros_factualidade,
    _erros_formula,
    _linha_contato,
    generate_cv_pipeline,
)
from app.llm import LLMUnavailable
from app.schemas import GenerateCvRequest


class GenerateCvTest(unittest.TestCase):
    def setUp(self):
        self.req = GenerateCvRequest.model_validate(
            {
                "perfilMestre": {
                    "nome": "Pessoa Teste",
                    "contato": {
                        "telefone": "+55 85 99999-0000",
                        "email": "pessoa@example.com",
                        "linkedin": "https://linkedin.com/in/pessoa",
                    },
                    "resumo": "Desenvolvedora backend com experiência em APIs REST e sistemas em produção.",
                    "experiencias": [
                        {
                            "empresa": "Empresa A",
                            "cargo": "Desenvolvedora Backend",
                            "periodo": "Jun. 2024 a atual",
                            "descricao": "- Atuei no desenvolvimento de APIs REST com Python e FastAPI em produção.\n- Contribuí para filas assíncronas com Redis junto ao time.",
                            "tecnologias": ["Python", "FastAPI", "Redis"],
                        }
                    ],
                    "formacao": ["Universidade A | ADS | 02/2023 - 12/2025"],
                    "certificacoes": ["Python, Escola A, 2025"],
                    "idiomas": ["Português, nativo", "Inglês, intermediário"],
                    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
                },
                "vaga": {
                    "titulo": "Desenvolvedora Backend Python",
                    "empresa": "Empresa B",
                    "descricao": "Buscamos Python, FastAPI, PostgreSQL e Docker para APIs REST.",
                },
                "keywords": [
                    {"termo": "Python", "peso": 1},
                    {"termo": "FastAPI", "peso": 0.9},
                    {"termo": "PostgreSQL", "peso": 0.8},
                    {"termo": "Docker", "peso": 0.7},
                ],
            }
        )

    @patch("app.generate.complete_model", side_effect=LLMUnavailable("offline"))
    def test_fallback_respeita_contrato_editorial(self, _complete):
        resultado = generate_cv_pipeline(self.req)
        markdown = resultado.markdown

        self.assertEqual([], _erros_contrato(markdown, self.req))
        self.assertTrue(markdown.startswith("# Pessoa Teste\n**Desenvolvedora Backend Python**"))
        self.assertIn("- Linguagens: Python", markdown)
        self.assertIn("**Empresa A** | Desenvolvedora Backend | 06/2024 - atual", markdown)
        self.assertNotIn("Descrição:", markdown)
        self.assertNotIn("usando .", markdown)
        self.assertIn("## CERTIFICAÇÕES\n- Python, Escola A, 2025", markdown)
        self.assertIn("Português, nativo | Inglês, intermediário", markdown)
        self.assertIn("[pessoa@example.com](mailto:pessoa@example.com)", markdown)
        self.assertIn("(https://linkedin.com/in/pessoa)", markdown)

    def test_linha_contato_resolve_link_real(self):
        contato = {
            "email": "pessoa@example.com",
            "linkedin": "linkedin.com/in/pessoa",
            "github": "github.com/pessoa",
            "telefone": "+55 85 99999-0000",
        }
        linha = _linha_contato(contato)

        self.assertIn("[pessoa@example.com](mailto:pessoa@example.com)", linha)
        self.assertIn("[linkedin.com/in/pessoa](https://linkedin.com/in/pessoa)", linha)
        self.assertIn("[github.com/pessoa](https://github.com/pessoa)", linha)
        self.assertIn("+55 85 99999-0000", linha)
        self.assertNotIn("[+55 85 99999-0000]", linha)

    def test_erros_formula_rejeita_vazamento_do_vocabulario_interno(self):
        self.assertEqual([], _erros_formula("Aumentei vendas em 30% usando Python."))
        self.assertTrue(_erros_formula("Resultado: aumento de 30% nas vendas."))
        self.assertTrue(_erros_formula("Entreguei o projeto usando ferramenta por extenso Python."))
        self.assertTrue(_erros_formula("Segui verbo de acao, resultado real e entrega."))

    def test_erros_coerencia_rejeita_resumo_desalinhado_da_vaga(self):
        markdown = (
            "# Pessoa Teste\n**Desenvolvedora Backend Python**\n"
            "telefone: 11999999999\n\n"
            "## RESUMO PROFISSIONAL\n"
            "Apaixonada por cozinhar e viajar nas horas vagas.\n\n"
            "## COMPETÊNCIAS\n- Linguagens: Python\n"
        )
        self.assertTrue(_erros_coerencia(markdown, self.req))

        markdown_coerente = markdown.replace(
            "Apaixonada por cozinhar e viajar nas horas vagas.",
            "Desenvolvedora backend Python com foco em APIs FastAPI e PostgreSQL.",
        )
        self.assertEqual([], _erros_coerencia(markdown_coerente, self.req))

    def test_antialucinacao_e_multiusuario_data_driven(self):
        sem_django = self.req.model_copy(deep=True)
        sem_django.vaga.titulo = "Desenvolvedora Python Django Next.js"
        sem_django.vaga.descricao = "Vaga com Python, Django, Next.js, APIs REST e Docker."
        sem_django.keywords = [
            {"termo": "Python", "peso": 1},
            {"termo": "Django", "peso": 1},
            {"termo": "Next.js", "peso": 1},
            {"termo": "Docker", "peso": 0.8},
        ]

        com_django = sem_django.model_copy(deep=True)
        com_django.perfil_mestre.skills.append("Django")
        com_django.perfil_mestre.experiencias[0].descricao += (
            "\n- Desenvolvi APIs REST com Django REST Framework e PostgreSQL."
        )

        erros_sem = _erros_factualidade(
            "Experiencia com Python, Django, Next.js e Docker.",
            sem_django,
        )
        erros_com = _erros_factualidade(
            "Experiencia com Python, Django e Docker.",
            com_django,
        )

        self.assertIn("django", erros_sem[0].lower())
        self.assertEqual([], erros_com)


if __name__ == "__main__":
    unittest.main()
