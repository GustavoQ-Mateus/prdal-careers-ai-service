import unittest
from unittest.mock import patch

from app.copiloto import redigir_formulario, redigir_mensagem
from app.generate import MARCADOR_CONTATO, _limpar_markdown, generate_cv_pipeline
from app.schemas import (
    GenerateCvRequest,
    GenerateCvResponse,
    RedigirFormularioRequest,
    RedigirFormularioResponse,
    RedigirMensagemRequest,
    RedigirMensagemResponse,
    RespostaFormulario,
)

EMAIL = "pessoa.privada@exemplo.dev"
TELEFONE = "+55 85 99999-1234"
ENDERECO = "Rua das Flores, 123, Fortaleza"
DADOS_PESSOAIS = (EMAIL, TELEFONE, "99999-1234", ENDERECO, "Rua das Flores")

PERFIL = {
    "nome": "Pessoa Candidata",
    "contato": {"email": EMAIL, "telefone": TELEFONE, "endereco": ENDERECO},
    "resumo": "Desenvolvedora back-end com APIs REST em Python.",
    "experiencias": [
        {
            "empresa": "Empresa A",
            "cargo": "Desenvolvedora Back-end",
            "periodo": "01/2023 - atual",
            "descricao": "- Desenvolvi APIs REST com Python e FastAPI em producao.",
            "tecnologias": ["Python", "FastAPI"],
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
        complete.return_value = GenerateCvResponse(
            markdown=f"# Pessoa Candidata\n**Desenvolvedora Python**\n{MARCADOR_CONTATO}\n\n## RESUMO\nTexto."
        )
        generate_cv_pipeline(_req())
        self.assertGreater(complete.call_count, 0)
        for chamada in complete.call_args_list:
            sistema, usuario = chamada.args[0], chamada.args[1]
            self.assertSemDadoPessoal(sistema, usuario)
            self.assertIn("Pessoa Candidata", usuario)
            self.assertIn(MARCADOR_CONTATO, sistema)

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
    def test_marcador_vira_linha_de_contato_do_perfil(self):
        markdown = _limpar_markdown(
            f"# Pessoa Candidata\n**Desenvolvedora Python**\n{MARCADOR_CONTATO}\n\n## RESUMO\nTexto.", _req()
        )
        linhas = [linha for linha in markdown.splitlines() if linha.strip()]
        self.assertIn(EMAIL, linhas[2])
        self.assertIn(TELEFONE, linhas[2])
        self.assertNotIn(MARCADOR_CONTATO, markdown)

    def test_contato_omitido_pelo_modelo_e_inserido(self):
        markdown = _limpar_markdown("# Pessoa Candidata\n**Desenvolvedora Python**\n\n## RESUMO\nTexto.", _req())
        linhas = [linha for linha in markdown.splitlines() if linha.strip()]
        self.assertIn(EMAIL, linhas[2])
        self.assertTrue(linhas[3].startswith("## "))

    def test_sem_contato_no_perfil_o_marcador_some(self):
        req = _req()
        req.perfil_mestre.contato = {}
        markdown = _limpar_markdown(
            f"# Pessoa Candidata\n**Desenvolvedora Python**\n{MARCADOR_CONTATO}\n\n## RESUMO\nTexto.", req
        )
        self.assertNotIn(MARCADOR_CONTATO, markdown)


if __name__ == "__main__":
    unittest.main()
