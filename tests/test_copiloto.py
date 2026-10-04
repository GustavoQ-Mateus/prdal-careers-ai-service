import json
import unittest

from fastapi.testclient import TestClient

from app.copiloto import (
    SYSTEM_TURNO,
    _catalogo,
    _narracao_ats_concluida,
    _regerar_por_perfil_atualizado,
    _texto_para_candidato,
    planejar_turno,
    redigir_formulario,
    redigir_mensagem,
)
from app.llm import LLMUnavailable
from app.main import app
from app.schemas import (
    MensagemTurno,
    RedigirFormularioRequest,
    RedigirMensagemRequest,
    ToolSpec,
    TurnRequest,
)
from tests.cliente_falso import ComClienteFalso, resposta


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
        self.assertIn("'Etapa 1: Aderencia do perfil-mestre'", SYSTEM_TURNO)
        self.assertIn("Aderencia do curriculo gerado'", SYSTEM_TURNO)
        self.assertNotIn("\u2014", SYSTEM_TURNO)
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
        self.assertNotIn("/curriculos/1", texto)

    def test_preserva_vocabulario_legitimo_do_candidato(self):
        frase = (
            "A vaga pede experiencia com JSON, APIs REST e tools de observabilidade; "
            "a rota de carreira e backend."
        )
        self.assertEqual(_texto_para_candidato(frase), frase)
        self.assertEqual(
            _texto_para_candidato("O payload da API e enviado em json."),
            "O payload da API e enviado em json.",
        )

    def test_remove_nomes_exatos_das_tools_recebidas(self):
        req = TurnRequest(tools=[ToolSpec(nome="consultar_agenda_extra", efeito="leitura")])
        texto = _texto_para_candidato("Usei consultar_agenda_extra e buscar_curriculo agora.", req)
        self.assertNotIn("consultar_agenda_extra", texto)
        self.assertNotIn("buscar_curriculo", texto)
        self.assertIn("agora", texto)


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
        self.assertIn("Pontos de atenção: secao obrigatoria ausente", resposta.texto)
        self.assertIn("[[NARRACAO_ATS_ETAPA_3]]", resposta.texto)
        self.assertIn("Etapa 1: Aderência do perfil-mestre", resposta.texto)
        self.assertIn("Etapa 3: Aderência do currículo gerado", resposta.texto)
        self.assertIn("Score: 76", resposta.texto)
        self.assertIn("Keywords ainda ausentes: Docker", resposta.texto)
        for proibida in ("aumentou", "melhorou", "reduziu", "\u2014"):
            self.assertNotIn(proibida, resposta.texto)


class RespostaVaziaCopilotoTest(unittest.TestCase):
    TURNO = TurnRequest(
        mensagens=[MensagemTurno(papel="user", conteudo="oi")],
        tools=[ToolSpec(nome="ler_perfil", efeito="leitura")],
    )

    def test_mensagem_vazia_tenta_um_reparo_e_vira_indisponibilidade(self):
        vazia = {"titulo": "t", "texto": "  ", "destino": ""}
        with ComClienteFalso(resposta(vazia), resposta(vazia)) as cliente:
            with self.assertRaises(LLMUnavailable):
                redigir_mensagem(RedigirMensagemRequest())
        self.assertEqual(2, len(cliente.requisicoes))
        self.assertIn("vazia", cliente.requisicoes[1]["messages"][2]["content"][0]["text"])

    def test_formulario_sem_respostas_vira_indisponibilidade(self):
        vazio = {"titulo": "t", "respostas": [], "texto": ""}
        with ComClienteFalso(resposta(vazio), resposta(vazio)):
            with self.assertRaises(LLMUnavailable):
                redigir_formulario(RedigirFormularioRequest(campos=["Por que esta vaga?"]))

    def test_turno_com_tipo_invalido_vira_indisponibilidade(self):
        outro = {"tipo": "outro", "texto": "oi", "tool": None, "argsJson": None}
        with ComClienteFalso(resposta(outro), resposta(outro)) as cliente:
            with self.assertRaises(LLMUnavailable):
                planejar_turno(self.TURNO)
        self.assertEqual(2, len(cliente.requisicoes))

    def test_turno_vazio_vira_indisponibilidade(self):
        vazio = {"tipo": "texto", "texto": "", "tool": None, "argsJson": None}
        with ComClienteFalso(resposta(vazio), resposta(vazio)):
            with self.assertRaises(LLMUnavailable):
                planejar_turno(self.TURNO)

    def test_tool_fora_do_catalogo_e_reparada(self):
        fora = {"tipo": "tool_call", "texto": None, "tool": "apagar_tudo", "argsJson": "{}"}
        valida = {"tipo": "tool_call", "texto": None, "tool": "ler_perfil", "argsJson": "{\"secao\": \"skills\"}"}
        with ComClienteFalso(resposta(fora), resposta(valida)) as cliente:
            turno = planejar_turno(self.TURNO)
        self.assertEqual("ler_perfil", turno.tool)
        self.assertEqual({"secao": "skills"}, turno.args)
        self.assertIn("fora do catalogo", cliente.requisicoes[1]["messages"][2]["content"][0]["text"])

    def test_args_que_nao_sao_objeto_sao_reparados(self):
        lista = {"tipo": "tool_call", "texto": None, "tool": "ler_perfil", "argsJson": "[1, 2]"}
        valida = {"tipo": "tool_call", "texto": None, "tool": "ler_perfil", "argsJson": "{}"}
        with ComClienteFalso(resposta(lista), resposta(valida)):
            turno = planejar_turno(self.TURNO)
        self.assertEqual({}, turno.args)

    def test_endpoint_de_redacao_responde_503_com_frase_de_produto(self):
        vazia = {"titulo": "t", "texto": "", "destino": ""}
        with ComClienteFalso(resposta(vazia), resposta(vazia)):
            resposta_http = TestClient(app).post("/copiloto/redigir-mensagem", json={})
        self.assertEqual(resposta_http.status_code, 503)
        self.assertIn("indisponível", resposta_http.json()["detail"])
        self.assertNotIn("vazia", resposta_http.json()["detail"])

    def test_endpoint_de_turno_responde_503(self):
        vazio = {"tipo": "texto", "texto": "", "tool": None, "argsJson": None}
        with ComClienteFalso(resposta(vazio), resposta(vazio)):
            resposta_http = TestClient(app).post(
                "/copiloto/turn", json={"mensagens": [{"papel": "user", "conteudo": "oi"}]}
            )
        self.assertEqual(resposta_http.status_code, 503)


if __name__ == "__main__":
    unittest.main()
