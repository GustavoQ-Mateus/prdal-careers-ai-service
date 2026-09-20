import unittest
from unittest.mock import patch

from app.generate import (
    _deterministic_request,
    _apenas_erro_formato_mecanico,
    _erros_completude,
    _erros_contrato,
    _erros_coerencia,
    _erros_factualidade,
    _erros_formula,
    _erros_metricas,
    _erros_ordem,
    _erros_saida,
    _limpar_markdown,
    _linha_contato,
    _normalizar_cabecalho_experiencia,
    _user,
    analisar_ats,
    generate_cv_pipeline,
    reduzir_curriculo,
)
from app.llm import LLMUnavailable
from app.schemas import GenerateCvRequest, GenerateCvResponse


def _req_tres_experiencias() -> GenerateCvRequest:
    return GenerateCvRequest.model_validate(
        {
            "perfilMestre": {
                "nome": "Gustavo Queiroz Mateus",
                "contato": {
                    "telefone": "+55 85 99120-7171",
                    "email": "gustavoqueirozunifor@edu.unifor.br",
                    "linkedin": "https://linkedin.com/in/gustavo-queiroz-mateus-935255283",
                },
                "resumo": "Desenvolvedor back-end com experiência em APIs REST e sistemas em produção.",
                "experiencias": [
                    {
                        "empresa": "Modera Road Inspector",
                        "cargo": "Desenvolvedor Full-Stack",
                        "periodo": "06/2026 - atual",
                        "descricao": (
                            "- Atuei no back-end de plataforma web em produção com Python (FastAPI) e PostgreSQL.\n"
                            "- Implementei autenticação JWT multi-tenant e filas assíncronas.\n"
                            "- Atuei na infraestrutura como código e CI/CD junto ao time."
                        ),
                        "tecnologias": ["Python", "FastAPI", "PostgreSQL"],
                    },
                    {
                        "empresa": "Saraiva Leão · Assessoria e Cálculos Judiciais",
                        "cargo": "Desenvolvedor Full-Stack",
                        "periodo": "03/2025 - atual",
                        "descricao": (
                            "- Construí um ERP corporativo com Python (FastAPI) e MySQL em produção.\n"
                            "- Implementei autenticação JWT multiempresa com auditoria.\n"
                            "- Assumi o deploy e a sustentação em produção."
                        ),
                        "tecnologias": ["Python", "FastAPI", "MySQL"],
                    },
                    {
                        "empresa": "Micro&Money · Softwares Inteligentes",
                        "cargo": "Estágio Full-Stack",
                        "periodo": "01/2026 - 04/2026",
                        "descricao": "- Atuei em módulos ERP com Java (Spring Boot) sobre MySQL.",
                        "tecnologias": ["Java", "Spring Boot", "MySQL"],
                    },
                ],
                "formacao": ["UNIFOR | ADS | 02/2025 - 06/2027"],
                "certificacoes": [],
                "idiomas": ["Português, nativo", "Inglês, intermediário"],
                "skills": ["Java", "Spring Boot", "Python", "FastAPI", "PostgreSQL", "MySQL"],
            },
            "vaga": {
                "titulo": "Desenvolvedor Back-End Java Jr",
                "empresa": "FCamara",
                "descricao": "Buscamos Java, Spring Boot, APIs REST e MySQL para sistemas corporativos.",
            },
            "keywords": [
                {"termo": "Java", "peso": 1},
                {"termo": "Spring Boot", "peso": 0.9},
                {"termo": "MySQL", "peso": 0.8},
                {"termo": "APIs REST", "peso": 0.7},
            ],
        }
    )


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

    def test_analise_avulsa_reaproveita_a_etapa_inicial_sem_gerar_curriculo(self):
        resultado = analisar_ats(self.req)

        self.assertIsInstance(resultado.score, int)
        self.assertIsInstance(resultado.keywords_encontradas, list)
        self.assertIsInstance(resultado.keywords_criticas_ausentes, list)
        self.assertTrue(resultado.veredicto)

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


class IntegridadeConteudoTest(unittest.TestCase):
    def setUp(self):
        self.req = _req_tres_experiencias()

    def _experiencia(self, ordem: list[str]) -> str:
        headers = {
            "modera": "**Modera Road Inspector** | Desenvolvedor Full-Stack | 06/2026 - atual\n- Atuei no back-end com Python (FastAPI) e PostgreSQL.",
            "saraiva": "**Saraiva Leão · Assessoria e Cálculos Judiciais** | Desenvolvedor Full-Stack | 03/2025 - atual\n- Construí um ERP com FastAPI e MySQL.",
            "micro": "**Micro&Money · Softwares Inteligentes** | Estágio Full-Stack | 01/2026 - 04/2026\n- Atuei em módulos ERP com Java (Spring Boot).",
        }
        corpo = "\n".join(headers[chave] for chave in ordem)
        return f"## EXPERIÊNCIA PROFISSIONAL\n{corpo}\n\n## FORMAÇÃO ACADÊMICA\n"

    def test_deterministic_inclui_as_tres_experiencias_em_ordem(self):
        markdown = _deterministic_request(self.req)

        self.assertIn("Modera Road Inspector", markdown)
        self.assertIn("Saraiva Leão", markdown)
        self.assertIn("Micro&Money", markdown)
        pos_modera = markdown.index("Modera Road Inspector")
        pos_saraiva = markdown.index("Saraiva Leão")
        pos_micro = markdown.index("Micro&Money")
        self.assertLess(pos_modera, pos_saraiva)
        self.assertLess(pos_saraiva, pos_micro)
        self.assertEqual([], _erros_saida(markdown, self.req))

    def test_completude_pega_experiencia_omitida(self):
        completo = self._experiencia(["modera", "saraiva", "micro"])
        self.assertEqual([], _erros_completude(completo, self.req))

        sem_modera = self._experiencia(["saraiva", "micro"])
        erros = _erros_completude(sem_modera, self.req)
        self.assertTrue(erros)
        self.assertIn("Modera Road Inspector", erros[0])

    def test_ordem_pega_inversao_cronologica(self):
        certo = self._experiencia(["modera", "saraiva", "micro"])
        self.assertEqual([], _erros_ordem(certo, self.req))

        invertido = self._experiencia(["micro", "modera", "saraiva"])
        self.assertTrue(_erros_ordem(invertido, self.req))

    def test_pipeline_recupera_omissao_do_modelo(self):
        omitido = GenerateCvResponse(
            markdown=(
                "# Gustavo Queiroz Mateus\n**Desenvolvedor Back-End Java Jr**\n"
                "+55 85 99120-7171\n\n"
                "## RESUMO PROFISSIONAL\nDesenvolvedor Java back-end.\n\n"
                "## COMPETÊNCIAS\n- Linguagens: Java\n\n"
                "## EXPERIÊNCIA PROFISSIONAL\n"
                "**Micro&Money · Softwares Inteligentes** | Estágio Full-Stack | 01/2026 - 04/2026\n"
                "- Atuei em módulos ERP com Java (Spring Boot).\n\n"
                "## FORMAÇÃO ACADÊMICA\nUNIFOR\n\n"
                "## CERTIFICAÇÕES\n\n## IDIOMAS\nPortuguês, nativo\n"
            )
        )
        with patch("app.generate.complete_model", return_value=omitido):
            resultado = generate_cv_pipeline(self.req)

        self.assertIsNotNone(resultado.degradacao)
        self.assertIn("Modera Road Inspector", resultado.markdown)
        self.assertIn("Saraiva Leão", resultado.markdown)
        self.assertIn("Micro&Money", resultado.markdown)
        pos_modera = resultado.markdown.index("Modera Road Inspector")
        pos_micro = resultado.markdown.index("Micro&Money")
        self.assertLess(pos_modera, pos_micro)

    def test_reduzir_aplica_corte_valido(self):
        reduzido = GenerateCvResponse(markdown=_deterministic_request(self.req))
        with patch("app.generate.complete_model", return_value=reduzido):
            resultado = reduzir_curriculo(self.req, "# markdown longo com duas paginas")

        self.assertIsNone(resultado.degradacao)
        self.assertIn("Modera Road Inspector", resultado.markdown)
        self.assertIn("Saraiva Leão", resultado.markdown)
        self.assertIn("Micro&Money", resultado.markdown)

    def test_reduzir_mantem_versao_anterior_quando_corte_falha(self):
        anterior = "# versao anterior que estoura pagina"
        with patch("app.generate.complete_model", side_effect=LLMUnavailable("offline")):
            resultado = reduzir_curriculo(self.req, anterior)

        self.assertEqual(anterior, resultado.markdown)
        self.assertIsNotNone(resultado.degradacao)

    def test_limpar_normaliza_hifen_nao_separavel(self):
        markdown = (
            "# Gustavo Queiroz Mateus\n**Desenvolvedor Back‑End Java Jr**\n"
            "contato\n\n## RESUMO PROFISSIONAL\nCI‑CD e back‑end.\n"
        )
        limpo = _limpar_markdown(markdown, self.req)

        self.assertNotIn("‑", limpo)
        self.assertIn("Back-End", limpo)
        self.assertIn("CI-CD", limpo)


class NormalizacaoCabecalhoExperienciaTest(unittest.TestCase):
    """Casos reais capturados na ADR 0033 (llama-3.3-70b-instruct via OpenRouter)."""

    def setUp(self):
        self.req = _req_tres_experiencias()

    def test_cabecalho_com_quatro_campos_e_mes_abreviado_e_normalizado(self):
        linha_real = (
            "### **Desenvolvedor Full-Stack** | Modera Road Inspector | "
            "Jun. 2026 a atual | Pernambuco"
        )
        normalizado = _normalizar_cabecalho_experiencia(linha_real, self.req)

        self.assertEqual(
            "**Desenvolvedor Full-Stack** | Modera Road Inspector | 06/2026 - atual",
            normalizado,
        )

    def test_cabecalho_de_tres_campos_com_heading_continua_normalizado(self):
        linha = "### Saraiva Leao | Desenvolvedor Full-Stack | Mar. 2025 a atual"
        normalizado = _normalizar_cabecalho_experiencia(linha, self.req)

        self.assertEqual(
            "**Saraiva Leao** | Desenvolvedor Full-Stack | 03/2025 - atual",
            normalizado,
        )

    def test_pipeline_aceita_experiencia_de_quatro_campos_e_mes_abreviado(self):
        markdown = (
            "# Gustavo Queiroz Mateus\n**Desenvolvedor Back-End Java Jr**\n"
            "+55 85 99120-7171 | gustavoqueirozunifor@edu.unifor.br\n\n"
            "## RESUMO PROFISSIONAL\nDesenvolvedor back-end Java.\n\n"
            "## COMPETÊNCIAS\n- Linguagens: Java\n\n"
            "## EXPERIÊNCIA PROFISSIONAL\n"
            "### **Desenvolvedor Full-Stack** | Modera Road Inspector | "
            "Jun. 2026 a atual | Pernambuco\n"
            "- Atuei no back-end com Python (FastAPI) e PostgreSQL.\n\n"
            "## FORMAÇÃO ACADÊMICA\nUNIFOR\n\n"
            "## CERTIFICAÇÕES\n\n## IDIOMAS\nPortuguês, nativo\n"
        )
        limpo = _limpar_markdown(markdown, self.req)

        self.assertIn(
            "**Desenvolvedor Full-Stack** | Modera Road Inspector | 06/2026 - atual",
            limpo,
        )
        self.assertNotIn("Jun.", limpo)
        self.assertEqual([], _erros_contrato(limpo, self.req))

    def test_heading_duplicando_titulo_e_removido_antes_do_contato(self):
        markdown = (
            "# Gustavo Queiroz Mateus\n"
            "**Desenvolvedor BackEnd Java Jr**\n"
            "## **Desenvolvedor BackEnd Java Jr**\n"
            "https://github.com/GustavoQ-Mateus | "
            "[linkedin.com/in/gustavo-queiroz-mateus-935255283]"
            "(https://linkedin.com/in/gustavo-queiroz-mateus-935255283) | "
            "+55 85 99120-7171 | gustavoqueirozunifor@edu.unifor.br\n\n"
            "## RESUMO PROFISSIONAL\nDesenvolvedor back-end Java.\n\n"
            "## COMPETÊNCIAS\n- Linguagens: Java\n\n"
            "## EXPERIÊNCIA PROFISSIONAL\n"
            "**Micro&Money** | Estagio Full-Stack | 01/2026 - 04/2026\n"
            "- Atuei em modulos ERP com Java (Spring Boot).\n\n"
            "## FORMAÇÃO ACADÊMICA\nUNIFOR\n\n"
            "## CERTIFICAÇÕES\n\n## IDIOMAS\nPortuguês, nativo\n"
        )
        limpo = _limpar_markdown(markdown, self.req)
        linhas_uteis = [linha for linha in limpo.splitlines() if linha.strip()]

        self.assertNotIn("## **Desenvolvedor BackEnd Java Jr**", limpo)
        self.assertFalse(linhas_uteis[2].startswith("#"))
        self.assertEqual([], _erros_contrato(limpo, self.req))

    def test_titulo_duplicado_com_variacao_de_case_e_pontuacao_e_removido(self):
        markdown = (
            "# Gustavo Queiroz Mateus\n"
            "**Desenvolvedor BackEnd Java Jr**\n"
            "### desenvolvedor backend java jr.\n"
            "+55 85 99120-7171 | gustavoqueirozunifor@edu.unifor.br\n\n"
            "## RESUMO PROFISSIONAL\nDesenvolvedor back-end Java.\n\n"
            "## COMPETÊNCIAS\n- Linguagens: Java\n\n"
            "## EXPERIÊNCIA PROFISSIONAL\n"
            "**Micro&Money** | Estagio Full-Stack | 01/2026 - 04/2026\n"
            "- Atuei em modulos ERP com Java (Spring Boot).\n\n"
            "## FORMAÇÃO ACADÊMICA\nUNIFOR\n\n"
            "## CERTIFICAÇÕES\n\n## IDIOMAS\nPortuguês, nativo\n"
        )
        limpo = _limpar_markdown(markdown, self.req)
        linhas_uteis = [linha for linha in limpo.splitlines() if linha.strip()]

        self.assertFalse(linhas_uteis[2].startswith("#"))
        self.assertEqual([], _erros_contrato(limpo, self.req))


class RetryPrescritivoTest(unittest.TestCase):
    def setUp(self):
        self.req = _req_tres_experiencias()
        self.analise = analisar_ats(self.req)

    def test_apenas_erro_formato_mecanico_identifica_corretamente(self):
        self.assertTrue(_apenas_erro_formato_mecanico(["linha de contato ausente"]))
        self.assertTrue(
            _apenas_erro_formato_mecanico(
                ["cabecalho de experiencia invalido", "ordem de secoes invalida"]
            )
        )
        self.assertFalse(
            _apenas_erro_formato_mecanico(
                [
                    "linha de contato ausente",
                    "tecnologia sem fonte factual no perfil/contexto do usuario: django",
                ]
            )
        )
        self.assertFalse(_apenas_erro_formato_mecanico([]))

    def test_user_inclui_exemplo_quando_erro_e_apenas_formato_mecanico(self):
        prompt = _user(
            self.req, self.analise, [], erros=["cabecalho de experiencia invalido"]
        )
        self.assertIn("formato exigido", prompt)

    def test_user_nao_inclui_exemplo_quando_erro_de_conteudo(self):
        prompt = _user(
            self.req,
            self.analise,
            [],
            erros=["tecnologia sem fonte factual no perfil/contexto do usuario: django"],
        )
        self.assertNotIn("formato exigido", prompt)

    @patch("app.generate._erros_saida", return_value=["linha de contato ausente"])
    @patch("app.generate.complete_model")
    def test_pipeline_tenta_ate_3_vezes_para_erro_de_formato_mecanico(
        self, mock_complete, _erros
    ):
        mock_complete.return_value = GenerateCvResponse(markdown="# x\n**y**\nz\n")
        resultado = generate_cv_pipeline(self.req)

        self.assertEqual(3, mock_complete.call_count)
        self.assertIsNotNone(resultado.degradacao)

    @patch(
        "app.generate._erros_saida",
        return_value=["tecnologia sem fonte factual no perfil/contexto do usuario: django"],
    )
    @patch("app.generate.complete_model")
    def test_pipeline_mantem_2_tentativas_para_erro_de_conteudo(
        self, mock_complete, _erros
    ):
        mock_complete.return_value = GenerateCvResponse(markdown="# x\n**y**\nz\n")
        resultado = generate_cv_pipeline(self.req)

        self.assertEqual(2, mock_complete.call_count)
        self.assertIsNotNone(resultado.degradacao)


class ErrosMetricasTest(unittest.TestCase):
    def setUp(self):
        self.req = _req_tres_experiencias()

    def test_rejeita_percentual_inventado_sem_fonte_factual(self):
        markdown = "- Reduzi o tempo de processamento em cerca de ~40% usando Java."
        erros = _erros_metricas(markdown, self.req)

        self.assertTrue(erros)
        self.assertIn("40%", erros[0])

    def test_aceita_percentual_com_correspondencia_literal_no_contexto(self):
        req = self.req.model_copy(deep=True)
        req.contexto = ["Reduzi o tempo de processamento em 40% no ultimo trimestre."]
        markdown = "- Reduzi o tempo de processamento em 40% usando Java."

        self.assertEqual([], _erros_metricas(markdown, req))

    def test_aceita_multiplicador_presente_no_perfil_mestre(self):
        req = self.req.model_copy(deep=True)
        req.perfil_mestre.experiencias[0].descricao += (
            "\n- Aumentei a velocidade de resposta da API em 3x."
        )
        markdown = "- Aumentei a velocidade de resposta da API em 3x."

        self.assertEqual([], _erros_metricas(markdown, req))

    def test_sem_padrao_numerico_nao_gera_erro(self):
        self.assertEqual(
            [], _erros_metricas("- Atuei em modulos ERP com Java.", self.req)
        )


if __name__ == "__main__":
    unittest.main()
