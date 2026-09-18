import unittest

from app.copiloto import _catalogo, _regerar_por_perfil_atualizado, _texto_para_candidato
from app.schemas import MensagemTurno, ToolSpec, TurnRequest


class CatalogoCopilotoTest(unittest.TestCase):
    def test_expoe_regras_dos_argumentos_para_llm(self):
        req = TurnRequest(
            tools=[
                ToolSpec(
                    nome="definir_proximo_passo",
                    efeito="escrita",
                    descricao="Cria uma acao",
                    parametros={
                        "tipo": "enum: REVISAR_VAGA | ENVIAR_CANDIDATURA | OUTRO"
                    },
                )
            ]
        )

        catalogo = _catalogo(req)

        self.assertIn("tipo: enum: REVISAR_VAGA | ENVIAR_CANDIDATURA | OUTRO", catalogo)


class PerfilAtualizadoCopilotoTest(unittest.TestCase):
    def test_le_perfil_antes_de_regerar(self):
        resposta = _regerar_por_perfil_atualizado(
            TurnRequest(
                oportunidade_id="vaga-1",
                mensagens=[
                    MensagemTurno(papel="user", conteudo="Atualizei minhas competencias, tente novamente"),
                ],
            ),
        )

        self.assertIsNotNone(resposta)
        self.assertEqual(resposta.tool, "ler_perfil")

    def test_regera_apos_leitura_do_perfil_atualizado(self):
        resposta = _regerar_por_perfil_atualizado(
            TurnRequest(
                oportunidade_id="vaga-1",
                mensagens=[
                    MensagemTurno(papel="user", conteudo="Atualizei minhas competencias, tente novamente"),
                    MensagemTurno(papel="user", conteudo="Ja coloquei no perfil"),
                    MensagemTurno(papel="tool", tool="ler_perfil", conteudo="{}"),
                ],
            ),
        )

        self.assertIsNotNone(resposta)
        self.assertEqual(resposta.tool, "gerar_curriculo")
        self.assertEqual(resposta.args, {"oportunidadeId": "vaga-1"})

    def test_nao_regera_sem_pedido_de_nova_tentativa(self):
        resposta = _regerar_por_perfil_atualizado(
            TurnRequest(
                oportunidade_id="vaga-1",
                mensagens=[
                    MensagemTurno(papel="user", conteudo="Atualizei minhas competencias no perfil"),
                    MensagemTurno(papel="tool", tool="ler_perfil", conteudo="{}"),
                ],
            ),
        )

        self.assertIsNone(resposta)

    def test_remove_identificadores_internos_do_texto(self):
        texto = _texto_para_candidato("Vou chamar editar_curriculo via PUT /curriculos/1 com JSON.")

        self.assertNotIn("editar_curriculo", texto)
        self.assertNotIn("PUT", texto)
        self.assertNotIn("JSON", texto)


if __name__ == "__main__":
    unittest.main()
