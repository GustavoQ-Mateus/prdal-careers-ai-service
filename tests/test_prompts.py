import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import carregador_prompts
from app.carregador_prompts import PromptAusente, carregar_prompts, obter
from app.generate import generate_cv_pipeline
from app.llm import LLMUnavailable
from app.main import app
from app.schemas import GenerateCvRequest

INSTRUCOES_DE_FERRAMENTA_LOCAL = (".claude", "curriculos/", "gerar.py", "converter.py", "template.py", "CV_FONT_DELTA")


def _escrever(pasta: Path, nome: str, conteudo: str) -> None:
    (pasta / nome).write_text(conteudo, encoding="utf-8")


class CarregadorPromptsTest(unittest.TestCase):
    def tearDown(self):
        carregar_prompts()

    def test_prompt_de_reescrita_esta_versionado_com_id_e_versao(self):
        prompt = carregar_prompts()["reescrita"]
        self.assertEqual(prompt.id, "reescrita")
        self.assertGreaterEqual(prompt.versao, 1)
        self.assertEqual(prompt.rotulo, f"reescrita.v{prompt.versao}")
        self.assertEqual(len(prompt.sha256), 64)
        for trecho in INSTRUCOES_DE_FERRAMENTA_LOCAL:
            self.assertNotIn(trecho, prompt.texto)

    def test_sem_prompt_obrigatorio_o_carregamento_falha(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaises(PromptAusente):
                carregar_prompts(Path(pasta))

    def test_cabecalho_divergente_do_nome_e_rejeitado(self):
        with tempfile.TemporaryDirectory() as pasta:
            _escrever(Path(pasta), "reescrita.v1.md", "---\nid: reescrita\nversao: 2\n---\ntexto\n")
            with self.assertRaises(PromptAusente):
                carregar_prompts(Path(pasta))

    def test_maior_versao_prevalece(self):
        with tempfile.TemporaryDirectory() as pasta:
            _escrever(Path(pasta), "reescrita.v1.md", "---\nid: reescrita\nversao: 1\n---\nprimeira\n")
            _escrever(Path(pasta), "reescrita.v2.md", "---\nid: reescrita\nversao: 2\n---\nsegunda\n")
            carregar_prompts(Path(pasta))
            self.assertEqual(obter("reescrita").texto, "segunda")

    def test_sem_o_arquivo_o_app_nao_inicia(self):
        with tempfile.TemporaryDirectory() as pasta:
            with patch.object(carregador_prompts, "DIRETORIO_PROMPTS", Path(pasta)):
                with self.assertRaises(PromptAusente):
                    with TestClient(app):
                        pass

    @patch("app.main._aquecer_embeddings")
    def test_com_o_arquivo_o_app_inicia(self, _aquecer):
        with TestClient(app) as cliente:
            self.assertEqual(cliente.get("/health").status_code, 200)

    @patch("app.generate.complete_model", side_effect=LLMUnavailable("offline"))
    def test_geracao_informa_a_versao_do_prompt(self, _complete):
        req = GenerateCvRequest.model_validate(
            {
                "perfilMestre": {"nome": "Pessoa", "skills": ["Python"]},
                "vaga": {"titulo": "Dev", "descricao": "Python"},
                "keywords": [{"termo": "Python", "peso": 1}],
            }
        )
        resposta = generate_cv_pipeline(req)
        self.assertEqual(resposta.prompt_version, obter("reescrita").rotulo)
        self.assertEqual(resposta.model_dump(by_alias=True)["promptVersion"], obter("reescrita").rotulo)


if __name__ == "__main__":
    unittest.main()
