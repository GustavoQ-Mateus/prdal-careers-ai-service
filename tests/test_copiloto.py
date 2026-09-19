import json
import unittest

from app.copiloto import (
    SYSTEM_TURNO,
    _catalogo,
    _narracao_ats_concluida,
    _regerar_por_perfil_atualizado,
    _texto_para_candidato,
)
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

    def test_reserva_etapas_para_narracao_ats_e_separa_as_duas_mensagens(self):
        self.assertNotIn("Etapa 1, registrar", SYSTEM_TURNO)
        self.assertIn("'Etapa 1 — Analise ATS'", SYSTEM_TURNO)
        self.assertIn("'Etapa 3 — Score pos-geracao'", SYSTEM_TURNO)
        self.assertIn("[[NARRACAO_ATS_ETAPA_3]]", SYSTEM_TURNO)


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
        self.assertEqual(resposta.tool, "analisar_ats")
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


class NarracaoAtsCopilotoTest(unittest.TestCase):
    def test_narra_as_etapas_1_e_3_com_dados_do_curriculo_concluido(self):
        analise_inicial = {
            "score": 48,
            "keywordsEncontradas": ["TypeScript"],
            "keywordsCriticasAusentes": ["Docker"],
            "pontosEliminatorios": ["secao obrigatoria ausente"],
            "veredicto": "Requer ajuste antes da candidatura.",
        }
        resposta = _narracao_ats_concluida(
            TurnRequest(
                mensagens=[
                    MensagemTurno(
                        papel="tool",
                        tool="buscar_curriculo",
                        conteudo=json.dumps({
                            "analiseInicial": analise_inicial,
                            "analiseFinal": {**analise_inicial, "score": 76},
                        }),
                    )
                ]
            )
        )

        self.assertIsNotNone(resposta)
        self.assertIn("Etapa 1", resposta.texto)
        self.assertIn("Keywords encontradas: TypeScript", resposta.texto)
        self.assertIn("Pontos eliminatorios: secao obrigatoria ausente", resposta.texto)
        self.assertIn("[[NARRACAO_ATS_ETAPA_3]]", resposta.texto)
        self.assertIn("Etapa 3", resposta.texto)
        self.assertIn("Score final: 76", resposta.texto)


if __name__ == "__main__":
    unittest.main()
