import unittest
from unittest.mock import patch

from app.keywords import extract_keywords
from app.llm import LLMUnavailable
from app.schemas import GenerateCvRequest, KeywordLlm, KeywordsLlmResponse
from app.generate import KeywordsUnavailable, generate_cv_pipeline


class KeywordsIntegrityTest(unittest.TestCase):
    @patch("app.keywords.complete_model", side_effect=LLMUnavailable("offline"))
    def test_falha_nao_vira_contagem_de_frequencia(self, _complete):
        with self.assertRaises(LLMUnavailable):
            extract_keywords("como desenvolvimento backend sera voce missao")

    def test_geracao_sem_keywords_nao_produz_score(self):
        req = GenerateCvRequest.model_validate(
            {
                "perfilMestre": {"nome": "Pessoa", "resumo": "Backend"},
                "vaga": {"titulo": "Backend", "empresa": "Empresa", "descricao": "Vaga"},
                "keywords": [],
            }
        )
        with self.assertRaises(KeywordsUnavailable):
            generate_cv_pipeline(req)

    @patch(
        "app.keywords.complete_model",
        return_value=KeywordsLlmResponse(
            keywords=[
                KeywordLlm(termo="Java", peso=1, tipo="stack"),
                KeywordLlm(termo="Portugal", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="Reino Unido", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="Brasil", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="diversidade", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="respeito", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="ética", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="inovação", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="comunidade tech", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="expansão global", peso=1, tipo="dominio_negocio"),
                KeywordLlm(termo="Spring Boot", peso=0.9, tipo="stack"),
            ]
        ),
    )
    def test_vaga_fcamara_persiste_apenas_keywords_tecnicas(self, _complete):
        keywords = extract_keywords(
            "Desenvolvedor Back-End Java Jr. A FCamara atua no Brasil, Portugal e Reino Unido, "
            "com diversidade, respeito, ética, inovação, comunidade tech e expansão global. "
            "Requisitos: Java e Spring Boot."
        )
        self.assertEqual({"Java", "Spring Boot"}, {item.termo for item in keywords})
        self.assertTrue(all(item.tipo for item in keywords))


if __name__ == "__main__":
    unittest.main()
