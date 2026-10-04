import unittest
from unittest.mock import patch

from app.copiloto import redigir_formulario, redigir_mensagem
from app.generate import generate_cv_pipeline
from app.llm import LLMUnavailable
from app.schemas import (
    FraseFonte,
    GenerateCvRequest,
    ReescritaEstruturada,
    RedigirFormularioRequest,
    RedigirFormularioResponse,
    RedigirMensagemRequest,
    RedigirMensagemResponse,
    RespostaFormulario,
)

EMAIL = "pessoa.privada@exemplo.dev"
TELEFONE = "+55 85 99999-1234"
LINKEDIN = "linkedin.com/in/pessoa-privada"
DADOS_PESSOAIS = (EMAIL, "99999-1234", LINKEDIN, "Fortaleza")

PERFIL = {
    "nome": "Pessoa Candidata",
    "emails": [{"valor": "outro@exemplo.dev", "principal": False}, {"valor": EMAIL, "principal": True}],
    "telefones": [{"ddi": "+55", "numero": "85 99999-1234", "principal": True}],
    "links": [{"tipo": "linkedin", "url": LINKEDIN}],
    "endereco": {"pais": "Brasil", "estado": "CE", "cidade": "Fortaleza"},
    "resumo": "Desenvolvedora back-end com APIs REST em Python.",
    "experiencias": [
        {
            "empresa": "Empresa A",
            "cargo": "Desenvolvedora Back-end",
            "dataInicioMes": 1, "dataInicioAno": 2023, "atual": True,
            "descricao": "- Desenvolvi APIs REST com Python e FastAPI em producao.",
        }
    ],
    "skills": ["Python", "FastAPI"],
}
VAGA = {"titulo": "Desenvolvedora Python", "empresa": "Empresa B", "descricao": "Python e FastAPI"}


def _req() -> GenerateCvRequest:
    return GenerateCvRequest.model_validate(
        {"perfilMestre": PERFIL, "vaga": VAGA, "keywords": [{"termo": "Python", "peso": 1}]}
    )


class PromptSemContatoTest(unittest.TestCase):
    def assertSemDadoPessoal(self, *textos: str):
        for texto in textos:
            for dado in DADOS_PESSOAIS:
                self.assertNotIn(dado, texto)

    @patch("app.generate.complete_model")
    def test_geracao_envia_perfil_sem_contato(self, complete):
        complete.return_value = ReescritaEstruturada(
            titulo=FraseFonte(texto="Desenvolvedora Back-end", fontes=["experiencia-1"]),
            resumo=[FraseFonte(texto="Desenvolvedora back-end com APIs REST em Python.", fontes=["resumo"])],
            experiencias=[],
            competencias=[],
            reparos=[],
        )
        generate_cv_pipeline(_req())
        self.assertEqual(1, complete.call_count)
        for chamada in complete.call_args_list:
            sistema, usuario = chamada.args[0], chamada.args[1]
            self.assertSemDadoPessoal(sistema, usuario)
            self.assertIn("Desenvolvi APIs REST com Python e FastAPI em producao.", usuario)

    @patch("app.copiloto.complete_model")
    def test_mensagem_ao_recrutador_sem_contato(self, complete):
        complete.return_value = RedigirMensagemResponse(titulo="t", texto="Ola", destino="email")
        redigir_mensagem(RedigirMensagemRequest.model_validate({"vaga": VAGA, "perfil": PERFIL, "contexto": ""}))
        sistema, usuario = complete.call_args.args[0], complete.call_args.args[1]
        self.assertSemDadoPessoal(sistema, usuario)
        self.assertNotIn("Contato:", usuario)
        self.assertIn("Pessoa Candidata", usuario)

    @patch("app.copiloto.complete_model")
    def test_respostas_de_formulario_sem_contato(self, complete):
        complete.return_value = RedigirFormularioResponse(
            titulo="t", respostas=[RespostaFormulario(campo="c", texto="r")], texto="r"
        )
        redigir_formulario(
            RedigirFormularioRequest.model_validate({"vaga": VAGA, "perfil": PERFIL, "campos": ["Por que voce?"]})
        )
        self.assertSemDadoPessoal(complete.call_args.args[0], complete.call_args.args[1])


class ContatoMontadoPorCodigoTest(unittest.TestCase):
    def _linhas(self, req):
        with patch("app.generate.complete_model", side_effect=LLMUnavailable("offline")):
            markdown = generate_cv_pipeline(req).markdown
        return [linha for linha in markdown.splitlines() if linha.strip()]

    def test_linha_de_contato_vem_do_perfil_e_so_com_os_principais(self):
        linhas = self._linhas(_req())
        self.assertIn(EMAIL, linhas[2])
        self.assertIn(TELEFONE, linhas[2])
        self.assertIn("Fortaleza - CE", linhas[2])
        self.assertNotIn("outro@exemplo.dev", linhas[2])

    def test_sem_contato_no_perfil_nao_ha_linha_de_contato(self):
        req = _req()
        req.perfil_mestre.emails = []
        req.perfil_mestre.telefones = []
        req.perfil_mestre.links = []
        req.perfil_mestre.endereco = None
        linhas = self._linhas(req)
        self.assertTrue(linhas[2].startswith("## "))


if __name__ == "__main__":
    unittest.main()
