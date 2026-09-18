import unittest

from app.copiloto import _catalogo
from app.schemas import ToolSpec, TurnRequest


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


if __name__ == "__main__":
    unittest.main()
