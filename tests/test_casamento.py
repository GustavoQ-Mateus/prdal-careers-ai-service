import unittest

from app.casamento import canonico, casar_termo, termo_presente
from app.fontes import fontes_da_geracao
from app.generate import _analise
from app.schemas import FraseFonte, GenerateCvRequest
from app.verificacao import motivo_da_rejeicao, termos_reconhecidos


def _req(keywords: list[str], skills: list[str]) -> GenerateCvRequest:
    return GenerateCvRequest.model_validate(
        {
            "perfilMestre": {"nome": "Pessoa", "skills": skills},
            "vaga": {"titulo": "Dev", "descricao": "vaga"},
            "keywords": [{"termo": termo, "peso": 1} for termo in keywords],
        }
    )


class CasamentoTest(unittest.TestCase):
    def test_termos_com_pontuacao_tecnica_sao_encontrados(self):
        texto = "Uso C#, .NET, Node.js, Next.js, CI/CD e C++ diariamente."
        for termo in ["C#", ".NET", "Node.js", "Next.js", "CI/CD", "C++"]:
            with self.subTest(termo=termo):
                self.assertTrue(termo_presente(termo, texto))

    def test_java_nao_e_encontrado_em_javascript(self):
        self.assertFalse(termo_presente("Java", "Experiencia com JavaScript e React."))

    def test_termo_composto_exige_componentes_adjacentes(self):
        self.assertFalse(termo_presente("Spring Boot", "sapato de boot"))
        self.assertFalse(termo_presente("Spring Boot", "Spring e mais tarde boot"))
        self.assertTrue(termo_presente("Spring Boot", "APIs com Spring Boot em producao"))
        self.assertTrue(termo_presente("Spring Boot", "APIs com SpringBoot"))

    def test_pontuacao_vizinha_nao_vira_termo_diferente(self):
        self.assertFalse(termo_presente("C", "Trabalhei com C# e F#."))
        self.assertFalse(termo_presente("JS", "Backend em Node.js"))
        self.assertFalse(termo_presente("Next", "Front com Next.js"))
        self.assertTrue(termo_presente("CSS", "Interfaces com HTML/CSS."))
        self.assertTrue(termo_presente("AWS", "Deploy na AWS."))

    def test_casos_de_borda_de_vagas_reais(self):
        self.assertTrue(termo_presente(".NET", "Experiencia com ASP.NET Core em producao"))
        self.assertFalse(termo_presente("R", "Salario de R$ 5 mil"))
        self.assertTrue(termo_presente("R", "Analise estatistica em R e Python"))
        self.assertTrue(termo_presente("C++", "uso C++17 no motor de jogo"))
        self.assertTrue(termo_presente("C#", "APIs em C#12 com .NET 8"))
        self.assertTrue(termo_presente("Vue.js", "Front em Vue 3 com Pinia"))
        self.assertFalse(termo_presente("C", "Trabalhei com C++17 e C#12."))
        self.assertFalse(termo_presente("Java", "Front em JavaScript"))

    def test_segunda_tentativa_compacta_aceita_variacao_de_grafia(self):
        self.assertTrue(termo_presente("Node.js", "Servicos em NodeJS"))
        self.assertTrue(termo_presente("CI/CD", "Pipelines de CI CD com GitHub Actions"))

    def test_sinonimos_canonicos_se_equivalem(self):
        pares = [
            ("JS", "JavaScript"),
            ("TS", "TypeScript"),
            ("Postgres", "PostgreSQL"),
            ("k8s", "Kubernetes"),
            ("React", "ReactJS"),
            ("React.js", "React"),
            ("Node", "Node.js"),
            ("NodeJS", "Node.js"),
            (".NET", "dotnet"),
            ("C#", "CSharp"),
            ("CI/CD", "CICD"),
        ]
        for termo, texto in pares:
            with self.subTest(termo=termo, texto=texto):
                self.assertTrue(termo_presente(termo, f"Atuei com {texto} em producao."))
                self.assertTrue(termo_presente(texto, f"Atuei com {termo} em producao."))
                self.assertEqual(canonico(termo), canonico(texto))

    def test_acentos_e_caixa_nao_importam(self):
        self.assertTrue(termo_presente("Integração Contínua", "integracao continua no time"))

    def test_ocorrencia_aponta_o_trecho_do_texto_original(self):
        texto = "Experiência com Spring Boot e Kafka"
        ocorrencia = casar_termo("Spring Boot", texto)
        self.assertIsNotNone(ocorrencia)
        self.assertEqual(texto[ocorrencia.inicio:ocorrencia.fim], "Spring Boot")

    def test_termo_vazio_nao_casa(self):
        self.assertIsNone(casar_termo("", "qualquer texto"))
        self.assertIsNone(casar_termo("   ", "qualquer texto"))

    def test_analise_e_factualidade_usam_o_mesmo_casamento(self):
        req = _req(["Java", "C#"], ["JavaScript", "C#"])
        analise = _analise("## RESUMO PROFISSIONAL\nUso JavaScript e C# no dia a dia.\n", req)
        self.assertEqual(analise.keywords_encontradas, ["C#"])
        self.assertIn("Java", analise.keywords_criticas_ausentes)
        fontes, termos = fontes_da_geracao(req), termos_reconhecidos(req)
        self.assertIn("Java", motivo_da_rejeicao(FraseFonte(texto="uso Java", fontes=["skills"]), fontes, termos))
        self.assertIsNone(motivo_da_rejeicao(FraseFonte(texto="uso C#", fontes=["skills"]), fontes, termos))

    def test_veredicto_descreve_cobertura_sem_prometer_filtro(self):
        req = _req(["Python", "Docker", "AWS", "Kafka"], [])
        casos = {
            "Python, Docker, AWS e Kafka": "alta",
            "Python e Docker": "média",
            "nada relacionado": "baixa",
        }
        for texto, nivel in casos.items():
            with self.subTest(nivel=nivel):
                veredicto = _analise(f"## RESUMO PROFISSIONAL\n{texto}\n", req).veredicto
                self.assertIn(f"Cobertura {nivel}", veredicto)
                for proibida in ("filtro", "elimin", "passa", "aprov"):
                    self.assertNotIn(proibida, veredicto.lower())


if __name__ == "__main__":
    unittest.main()
