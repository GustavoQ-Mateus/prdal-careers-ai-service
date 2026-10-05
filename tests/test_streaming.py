import json
import os
import random
import threading
import time
import unittest

import anthropic
import httpx
from fastapi.testclient import TestClient

from app.copiloto import SYSTEM_TURNO, SanitizadorDeStream, _nomes_protegidos, _texto_para_candidato, planejar_turno_em_stream
from app.llm import LLMUnavailable, OperacaoCancelada, StreamInterrompido, operacao
from app.main import app
from app.schemas import TurnRequest
from tests.cliente_falso import ComClienteFalso, StreamFalso, resposta_blocos

TOOL_PERFIL = {
    "name": "ler_perfil",
    "description": "Le o perfil-mestre do candidato",
    "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}


def turno(texto="oi") -> TurnRequest:
    return TurnRequest.model_validate(
        {"mensagens": [{"role": "user", "content": [{"type": "text", "text": texto}]}], "tools": [TOOL_PERFIL]}
    )


def falha_de_conexao():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def sanitizar_em_partes(texto, cortes):
    saida = []
    sanitizador = SanitizadorDeStream(saida.append, _nomes_protegidos(None))
    inicio = 0
    for corte in sorted(cortes):
        sanitizador.delta(texto[inicio:corte])
        inicio = corte
    sanitizador.delta(texto[inicio:])
    sanitizador.finalizar()
    return saida


class SanitizadorDeStreamTest(unittest.TestCase):
    def test_nome_partido_entre_deltas_e_substituido(self):
        saida = []
        sanitizador = SanitizadorDeStream(saida.append, _nomes_protegidos(None))
        sanitizador.delta("Vou consultar com ler_")
        self.assertEqual(["Vou consultar com"], saida)
        sanitizador.delta("perfil e depois buscar_curr")
        sanitizador.delta("iculo, sem JSON.")
        sanitizador.finalizar()
        texto = "".join(saida)
        self.assertNotIn("ler_perfil", texto)
        self.assertNotIn("buscar_curriculo", texto)
        self.assertEqual("Vou consultar com esta acao e depois esta acao, sem JSON.", texto)

    def test_rota_partida_entre_deltas_e_substituida(self):
        texto = "".join(sanitizar_em_partes("Chamei GET /curriculos/1 agora", [5, 8, 12, 20]))
        self.assertEqual("Chamei esta acao agora", texto)

    def test_palavra_que_so_comeca_igual_nao_e_retida_nem_trocada(self):
        texto = "".join(sanitizar_em_partes("O xler_perfil e ler_perfilado ficam; ler fica.", [3, 7, 17, 24, 40]))
        self.assertEqual("O xler_perfil e ler_perfilado ficam; ler fica.", texto)

    def test_saida_em_stream_igual_a_saida_inteira_para_qualquer_corte(self):
        textos = [
            "Use ler_perfil, depois analisar_ats e POST /oportunidades/1/gerar-cv; JSON e tools ficam.",
            "registrar_nota registrar_notas xregistrar_nota GET\n/x DELETE /a/b?c=1 fim",
            "Etapa 1: ler o perfil. gerar_curriculo! status_geracao? PUT  /y",
            "Fiz o curso de Docker \u2014 foi puxado. De 2019\u20132021 e 10 \u2013 20 vagas; full-stack \u2013 fim \u2014",
            "\u2014 item um\n\u2014 item dois \u2014 ler_perfil \u2014, e bem\u2014feito.",
        ]
        aleatorio = random.Random(7)
        for texto in textos:
            esperado = _texto_para_candidato(texto)
            for _ in range(300):
                cortes = aleatorio.sample(range(1, len(texto)), aleatorio.randint(1, min(12, len(texto) - 1)))
                self.assertEqual(esperado, "".join(sanitizar_em_partes(texto, cortes)), cortes)
            self.assertEqual(esperado, "".join(sanitizar_em_partes(texto, range(1, len(texto)))))


class TravessaoTest(unittest.TestCase):
    def test_travessao_vira_virgula_sem_tocar_hifen_nem_intervalo(self):
        casos = {
            "Terminei o curso de Docker \u2014 foi bem puxado.": "Terminei o curso de Docker, foi bem puxado.",
            "A vaga \u2013 de Python \u2013 pede SQL.": "A vaga, de Python, pede SQL.",
            "Atuei de 2019\u20132021 com 10 \u2013 20 pessoas.": "Atuei de 2019\u20132021 com 10 \u2013 20 pessoas.",
            "Perfil full-stack e back-end.": "Perfil full-stack e back-end.",
            "\u2014 primeiro\n\u2014 segundo": "- primeiro\n- segundo",
            "Pronto \u2014.": "Pronto.",
            "Pronto \u2014": "Pronto",
        }
        for texto, esperado in casos.items():
            self.assertEqual(esperado, _texto_para_candidato(texto), texto)

    def test_travessao_partido_entre_deltas_sai_igual_ao_texto_inteiro(self):
        texto = "Curso de Docker \u2014 foi puxado, de 2019\u20132021."
        for corte in range(1, len(texto)):
            self.assertEqual(_texto_para_candidato(texto), "".join(sanitizar_em_partes(texto, [corte])), corte)

    def test_turno_entrega_texto_final_sem_travessao(self):
        final = resposta_blocos([{"type": "text", "text": "Curso de Docker \u2014 parabens!"}])
        emitidos = []
        with ComClienteFalso(StreamFalso(["Curso de Docker ", "\u2014", " parabens!"], final)):
            res = planejar_turno_em_stream(turno(), emitidos.append)
        self.assertEqual("Curso de Docker, parabens!", "".join(emitidos))
        self.assertEqual("Curso de Docker, parabens!", res.conteudo[0]["text"])

    def test_system_do_turno_pede_para_nao_usar_travessao(self):
        self.assertIn("Nao use travessao nem meia-risca", SYSTEM_TURNO)


class TurnoEmStreamTest(unittest.TestCase):
    def test_primeiro_delta_sai_antes_do_fim_da_geracao(self):
        final = resposta_blocos([{"type": "text", "text": "Li seu perfil agora."}])
        stream = StreamFalso(["Li ", "seu ", "perfil ", "agora."], final, atraso=0.05)
        emitidos = []
        with ComClienteFalso(stream) as cliente:
            res = planejar_turno_em_stream(turno(), lambda texto: emitidos.append((time.monotonic(), texto)))
        self.assertLess(emitidos[0][0], stream.instantes[-1])
        self.assertGreater(len(emitidos), 1)
        self.assertEqual("Li seu perfil agora.", "".join(texto for _, texto in emitidos))
        self.assertEqual([{"type": "text", "text": "Li seu perfil agora."}], res.conteudo)
        self.assertTrue(stream.fechado)
        self.assertEqual("ler_perfil", cliente.requisicoes[0]["tools"][0]["name"])

    def test_tool_use_chega_no_fim_e_blocos_de_texto_ganham_separador(self):
        bloco = {"type": "tool_use", "id": "toolu_1", "name": "ler_perfil", "input": {}}
        final = resposta_blocos(
            [{"type": "text", "text": "Vou usar ler_perfil."}, {"type": "text", "text": "Pronto."}, bloco],
            stop_reason="tool_use",
        )
        stream = StreamFalso([["Vou usar ler_", "perfil."], ["Pronto."]], final)
        emitidos = []
        with ComClienteFalso(stream):
            res = planejar_turno_em_stream(turno(), emitidos.append)
        self.assertEqual("Vou usar esta acao.\n\nPronto.", "".join(emitidos))
        self.assertEqual("Vou usar esta acao.", res.conteudo[0]["text"])
        self.assertEqual(bloco, res.conteudo[2])
        self.assertEqual("tool_use", res.parada)

    def test_repete_so_antes_do_primeiro_delta(self):
        final = resposta_blocos([{"type": "text", "text": "Oi."}])
        with ComClienteFalso(StreamFalso(["Oi"], final, falha_apos=0, falha=falha_de_conexao()), StreamFalso(["Oi."], final)) as cliente:
            os.environ["AI_MAX_RETRIES"] = "1"
            emitidos = []
            with operacao() as op:
                planejar_turno_em_stream(turno(), emitidos.append)
        self.assertEqual(2, len(cliente.requisicoes))
        self.assertEqual(["Oi."], emitidos)
        self.assertEqual(2, op.uso.chamadas)

    def test_falha_depois_do_primeiro_delta_nao_repete_e_registra_uso_parcial(self):
        final = resposta_blocos([{"type": "text", "text": "nunca"}])
        stream = StreamFalso(["Comecei ", "a responder"], final, falha_apos=1, falha=falha_de_conexao(), entrada=321)
        with ComClienteFalso(stream, StreamFalso(["outra"], final)) as cliente:
            os.environ["AI_MAX_RETRIES"] = "2"
            emitidos = []
            with operacao() as op:
                with self.assertRaises(StreamInterrompido):
                    planejar_turno_em_stream(turno(), emitidos.append)
        self.assertEqual(1, len(cliente.requisicoes))
        self.assertEqual(["Comecei"], emitidos)
        self.assertEqual(321, op.uso.entrada)

    def test_prazo_esgotado_no_meio_encerra_o_stream(self):
        final = resposta_blocos([{"type": "text", "text": "a b c"}])
        stream = StreamFalso(["a ", "b ", "c"], final, atraso=0.08)
        with ComClienteFalso(stream):
            os.environ["AI_TIMEOUT_PISO_S"] = "0.01"
            try:
                with operacao(prazo_ms=100):
                    with self.assertRaises(LLMUnavailable):
                        planejar_turno_em_stream(turno(), lambda _texto: None)
            finally:
                os.environ.pop("AI_TIMEOUT_PISO_S", None)
        self.assertTrue(stream.fechado)


class CancelamentoTest(unittest.TestCase):
    def test_uso_parcial_sai_no_inicio_da_mensagem(self):
        final = resposta_blocos([{"type": "text", "text": "Oi."}])
        usos = []
        with ComClienteFalso(StreamFalso(["Oi."], final, entrada=777, uso_inicial=True)):
            with operacao():
                planejar_turno_em_stream(turno(), lambda _t: None, usos.append)
        self.assertEqual(1, len(usos))
        self.assertEqual(777, usos[0].entrada)
        self.assertEqual(1, usos[0].chamadas)

    def test_cancelamento_encerra_o_stream_sem_texto_e_guarda_o_uso(self):
        cancelado = threading.Event()
        final = resposta_blocos([{"type": "tool_use", "id": "t1", "name": "ler_perfil", "input": {}}], stop_reason="tool_use")
        stream = StreamFalso([[], []], final, entrada=432, ao_evento=cancelado.set)
        with ComClienteFalso(stream, StreamFalso([], final)) as cliente:
            os.environ["AI_MAX_RETRIES"] = "2"
            with operacao() as op:
                op.cancelada = cancelado
                with self.assertRaises(OperacaoCancelada):
                    planejar_turno_em_stream(turno(), lambda _t: None)
        self.assertEqual(1, len(cliente.requisicoes))
        self.assertTrue(stream.fechado)
        self.assertEqual(432, op.uso.entrada)


class EndpointDeStreamTest(unittest.TestCase):
    def linhas(self, resposta_http):
        return [json.loads(linha) for linha in resposta_http.text.splitlines() if linha.strip()]

    def test_endpoint_emite_deltas_e_fecha_com_blocos_e_uso(self):
        bloco = {"type": "tool_use", "id": "toolu_9", "name": "ler_perfil", "input": {}}
        final = resposta_blocos([{"type": "text", "text": "Vou ler."}, bloco], stop_reason="tool_use", cache_lida=900)
        with ComClienteFalso(StreamFalso(["Vou ", "ler."], final)):
            resposta_http = TestClient(app).post(
                "/copiloto/turn/stream",
                json={"mensagens": [{"role": "user", "content": [{"type": "text", "text": "oi"}]}], "tools": [TOOL_PERFIL]},
                headers={"X-Prdal-Prazo-Ms": "30000"},
            )
        self.assertEqual(200, resposta_http.status_code)
        self.assertTrue(resposta_http.headers["content-type"].startswith("application/x-ndjson"))
        linhas = self.linhas(resposta_http)
        self.assertEqual(["delta", "delta", "fim"], [linha["tipo"] for linha in linhas])
        self.assertEqual("Vou ler.", "".join(linha["texto"] for linha in linhas[:-1]))
        self.assertEqual(bloco, linhas[-1]["conteudo"][1])
        self.assertEqual("tool_use", linhas[-1]["parada"])
        self.assertEqual(900, linhas[-1]["uso"]["cacheLida"])

    def test_falha_no_meio_vira_linha_de_erro_com_uso(self):
        final = resposta_blocos([{"type": "text", "text": "x"}])
        stream = StreamFalso(["Parte ", "dois"], final, falha_apos=1, falha=falha_de_conexao(), entrada=50)
        with ComClienteFalso(stream):
            resposta_http = TestClient(app).post(
                "/copiloto/turn/stream", json={"mensagens": [{"role": "user", "content": [{"type": "text", "text": "oi"}]}]}
            )
        linhas = self.linhas(resposta_http)
        self.assertEqual(["delta", "erro"], [linha["tipo"] for linha in linhas])
        self.assertTrue(linhas[-1]["interrompido"])
        self.assertEqual(50, linhas[-1]["uso"]["entrada"])
        self.assertIn("indisponível", linhas[-1]["detail"])

    def test_endpoint_emite_o_uso_parcial_antes_dos_deltas(self):
        final = resposta_blocos([{"type": "text", "text": "Oi."}], entrada=640)
        with ComClienteFalso(StreamFalso(["Oi."], final, entrada=640, uso_inicial=True)):
            resposta_http = TestClient(app).post(
                "/copiloto/turn/stream", json={"mensagens": [{"role": "user", "content": [{"type": "text", "text": "oi"}]}]}
            )
        linhas = self.linhas(resposta_http)
        self.assertEqual(["uso", "delta", "fim"], [linha["tipo"] for linha in linhas])
        self.assertEqual(640, linhas[0]["uso"]["entrada"])
        self.assertEqual(640, linhas[-1]["uso"]["entrada"])

    def test_falha_antes_do_primeiro_delta_nao_e_interrompida(self):
        with ComClienteFalso(falha_de_conexao()):
            resposta_http = TestClient(app).post(
                "/copiloto/turn/stream", json={"mensagens": [{"role": "user", "content": [{"type": "text", "text": "oi"}]}]}
            )
        linhas = self.linhas(resposta_http)
        self.assertEqual(["erro"], [linha["tipo"] for linha in linhas])
        self.assertFalse(linhas[0]["interrompido"])


if __name__ == "__main__":
    unittest.main()
