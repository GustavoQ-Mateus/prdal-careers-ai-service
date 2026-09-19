import unittest
from unittest.mock import patch

from app.keywords import extract_keywords
from app.llm import LLMUnavailable
from app.schemas import GenerateCvRequest
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


if __name__ == "__main__":
    unittest.main()
