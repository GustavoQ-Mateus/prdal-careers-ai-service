import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import llm
from app.contexto import contador, linha_de_referencia, montar_contexto
from app.copiloto import planejar_turno
from app.schemas import TurnRequest
from tests.cliente_falso import resposta, resposta_blocos

TOOL = {
    "name": "buscar_oportunidade",
    "description": "Detalha uma oportunidade",
    "input_schema": {
        "type": "object",
        "properties": {"oportunidadeId": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}
SYSTEM = [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}]


def usuario(texto):
    return {"role": "user", "content": [{"type": "text", "text": texto}]}


def chamada(tool_id, nome="buscar_oportunidade"):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": nome, "input": {}}]}


def resultado(tool_id, conteudo):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": conteudo}]}


def resultado_grande(k):
    return json.dumps({"id": f"vaga-{k}", "titulo": f"Vaga {k}", "descricao": "Requisitos da vaga. " * 300})


class ClienteDeConversa:
    def __init__(self, contar=None):
        self.requisicoes = []
        self.resumos = 0
        self._contar = contar
        self.contagens = 0
        self.messages = self

    def with_options(self, **_):
        return self

    def create(self, **requisicao):
        self.requisicoes.append(copy.deepcopy(requisicao))
        if "tools" in requisicao:
            return resposta_blocos([{"type": "text", "text": f"Resposta {len(self.requisicoes)}."}], entrada=50)
        self.resumos += 1
        return resposta({"resumo": f"RESUMO-{self.resumos}: vaga-0 registrada e analisada."}, entrada=50)

    def count_tokens(self, **payload):
        if self._contar is None:
            raise AttributeError("count_tokens")
        self.contagens += 1
        return SimpleNamespace(input_tokens=self._contar(payload))


def pareado(mensagens):
    for i, mensagem in enumerate(mensagens):
        usos = [b["id"] for b in mensagem["content"] if b["type"] == "tool_use"]
        if not usos:
            continue
        seguinte = mensagens[i + 1]
        resultados = [b["tool_use_id"] for b in seguinte["content"] if b["type"] == "tool_result"]
        if sorted(usos) != sorted(resultados) or seguinte["content"][0]["type"] != "tool_result":
            return False
    return True


class Ambiente:
    def __init__(self, cliente, **env):
        self.cliente = cliente
        self.env = {"AI_MODEL": "claude-teste", "ANTHROPIC_API_KEY": "sk-teste", "AI_MAX_RETRIES": "0", **env}

    def __enter__(self):
        self._patch = patch.dict(os.environ, self.env)
        self._patch.start()
        self._anterior = llm._cliente
        llm.definir_cliente(self.cliente)
        contador.reiniciar()
        return self.cliente

    def __exit__(self, *_):
        llm.definir_cliente(self._anterior)
        self._patch.stop()
        contador.reiniciar()
        return False


class JanelaDeContextoTest(unittest.TestCase):
    ORCAMENTO = 6000

    def test_cinquenta_trocas_cabem_no_orcamento_com_resumo_persistido_e_pares_inteiros(self):
        persistidas = []
        resumo = None
        indice = 0
        with Ambiente(ClienteDeConversa(), AI_ORCAMENTO_ENTRADA_TOKENS=str(self.ORCAMENTO)) as cliente:
            for k in range(50):
                troca = [
                    usuario(f"Pergunta {k}: " + "detalhe da minha duvida sobre a vaga. " * 15),
                    chamada(f"toolu_{k}"),
                    resultado(f"toolu_{k}", resultado_grande(k)),
                ]
                persistidas.append({"indice": indice, "mensagens": troca})
                desde = resumo["ate"] if resumo else 0
                enviadas = [t for t in persistidas if t["indice"] >= desde]
                req = TurnRequest.model_validate(
                    {"oportunidadeId": "vaga-0", "trocas": enviadas, "resumo": resumo, "tools": [TOOL]}
                )
                antes = len(cliente.requisicoes)
                res = planejar_turno(req)
                turno = cliente.requisicoes[-1]
                carga = {"system": turno["system"], "tools": turno["tools"], "messages": turno["messages"]}
                self.assertLessEqual(contador.estimar(carga), self.ORCAMENTO, f"troca {k}")
                self.assertTrue(pareado(turno["messages"]), f"troca {k}")
                self.assertEqual("user", turno["messages"][0]["role"])
                ultimo = turno["messages"][-1]["content"][-1]
                self.assertIn("Requisitos da vaga.", ultimo["content"])
                vigente = res.resumo.texto if res.resumo else (resumo or {}).get("texto")
                if vigente:
                    self.assertIn(vigente, json.dumps(turno["messages"][0], ensure_ascii=False))
                if res.resumo:
                    self.assertIn(res.resumo.ate, [t["indice"] for t in persistidas])
                    self.assertGreater(res.resumo.ate, desde)
                    self.assertEqual(antes + 2, len(cliente.requisicoes))
                    resumo = res.resumo.model_dump()
                else:
                    self.assertEqual(antes + 1, len(cliente.requisicoes))
                troca.append({"role": "assistant", "content": res.conteudo})
                indice += 4
        self.assertGreaterEqual(cliente.resumos, 2)
        self.assertLess(cliente.resumos, 15)
        resumos = [r for r in cliente.requisicoes if "tools" not in r]
        self.assertTrue(all(r["output_config"]["effort"] == "low" for r in resumos))
        self.assertIn("RESUMO-1", resumos[1]["messages"][0]["content"][0]["text"])

    def test_resumo_persistido_e_reaproveitado_sem_nova_chamada(self):
        req = TurnRequest.model_validate(
            {
                "trocas": [{"indice": 8, "mensagens": [usuario("e agora?")]}],
                "resumo": {"texto": "Candidato registrou vaga-7.", "ate": 8},
                "tools": [TOOL],
            }
        )
        with Ambiente(ClienteDeConversa()) as cliente:
            res = planejar_turno(req)
        self.assertIsNone(res.resumo)
        self.assertEqual(1, len(cliente.requisicoes))
        primeira = cliente.requisicoes[0]["messages"][0]["content"]
        self.assertIn("Oportunidade em foco: nenhuma", primeira[0]["text"])
        self.assertIn("Candidato registrou vaga-7.", primeira[1]["text"])
        self.assertEqual("e agora?", primeira[2]["text"])

    def test_resultados_de_trocas_antigas_viram_linha_de_referencia(self):
        req = TurnRequest.model_validate(
            {
                "oportunidadeId": "vaga-1",
                "trocas": [
                    {"indice": 0, "mensagens": [usuario("ver"), chamada("t1"), resultado("t1", resultado_grande(1))]},
                    {"indice": 4, "mensagens": [usuario("de novo"), chamada("t2"), resultado("t2", resultado_grande(2))]},
                ],
                "tools": [TOOL],
            }
        )
        with Ambiente(ClienteDeConversa()):
            payload, novo = montar_contexto(req, SYSTEM, [TOOL])
        self.assertIsNone(novo)
        conteudos = [b["content"] for m in payload["messages"] for b in m["content"] if b["type"] == "tool_result"]
        self.assertIn("resultado antigo de buscar_oportunidade compactado; ids citados: vaga-1", conteudos[0])
        self.assertIn("Requisitos da vaga.", conteudos[1])
        self.assertIn("Oportunidade em foco: vaga-1", payload["messages"][0]["content"][0]["text"])

    def test_contagem_pela_api_e_usada_e_calibra_a_estimativa(self):
        req = TurnRequest.model_validate({"trocas": [{"indice": 0, "mensagens": [usuario("oi")]}], "tools": [TOOL]})
        cliente = ClienteDeConversa(contar=lambda payload: 10 * len(json.dumps(payload["messages"])))
        with Ambiente(cliente):
            antes = contador.fator
            payload, _ = montar_contexto(req, SYSTEM, [TOOL])
            self.assertEqual(1, cliente.contagens)
            self.assertGreater(contador.fator, antes)
            self.assertFalse(contador.api_indisponivel)

    def test_canal_sem_contagem_cai_na_estimativa(self):
        req = TurnRequest.model_validate({"trocas": [{"indice": 0, "mensagens": [usuario("oi")]}], "tools": [TOOL]})
        with Ambiente(ClienteDeConversa(contar=None)) as cliente:
            planejar_turno(req)
            self.assertTrue(contador.api_indisponivel)
        self.assertEqual(1, len(cliente.requisicoes))

    def test_falha_do_resumo_nao_persiste_e_ainda_respeita_o_orcamento(self):
        class SemResumo(ClienteDeConversa):
            def create(self, **requisicao):
                if "tools" not in requisicao:
                    self.requisicoes.append(requisicao)
                    raise llm.LLMUnavailable("fora")
                return super().create(**requisicao)

        trocas = [
            {"indice": 4 * k, "mensagens": [usuario("x" * 3000), chamada(f"t{k}"), resultado(f"t{k}", resultado_grande(k))]}
            for k in range(8)
        ]
        req = TurnRequest.model_validate({"trocas": trocas, "tools": [TOOL]})
        with Ambiente(SemResumo(), AI_ORCAMENTO_ENTRADA_TOKENS="5000"):
            payload, novo = montar_contexto(req, SYSTEM, [TOOL])
            self.assertLessEqual(contador.estimar(payload), 5000)
        self.assertIsNone(novo)
        self.assertTrue(pareado(payload["messages"]))

    def test_resultado_unico_maior_que_o_orcamento_e_encurtado_com_aviso(self):
        req = TurnRequest.model_validate(
            {
                "trocas": [{"indice": 0, "mensagens": [usuario("ver"), chamada("t1"), resultado("t1", "y" * 40000)]}],
                "tools": [TOOL],
            }
        )
        with Ambiente(ClienteDeConversa(), AI_ORCAMENTO_ENTRADA_TOKENS="3000"):
            payload, _ = montar_contexto(req, SYSTEM, [TOOL])
            self.assertLessEqual(contador.estimar(payload), 3000)
        conteudo = payload["messages"][-1]["content"][0]["content"]
        self.assertIn("conteudo encurtado para caber no orcamento", conteudo)
        self.assertTrue(conteudo.endswith("</dado_nao_confiavel>"))

    def test_mesmo_payload_e_contado_uma_vez_pela_api(self):
        req = TurnRequest.model_validate(
            {
                "trocas": [{"indice": 0, "mensagens": [usuario("ver"), chamada("t1"), resultado("t1", "y" * 40000)]}],
                "tools": [TOOL],
            }
        )
        contados = []

        def contar(payload):
            contados.append(json.dumps(payload, sort_keys=True))
            return len(json.dumps(payload)) // 3

        with Ambiente(ClienteDeConversa(contar=contar), AI_ORCAMENTO_ENTRADA_TOKENS="3000"):
            payload, _ = montar_contexto(req, SYSTEM, [TOOL])
            self.assertIn("conteudo encurtado", payload["messages"][-1]["content"][0]["content"])
            self.assertEqual(len(contados), len(set(contados)))
            self.assertGreaterEqual(len(contados), 2)
            antes = len(contados)
            contador.contar(payload)
            self.assertEqual(antes, len(contados))

    def test_linha_de_referencia_cita_ids_sem_repetir(self):
        linha = linha_de_referencia("listar_curriculos", json.dumps([{"id": "cv-1"}, {"id": "cv-1", "vagaId": "v-2"}]))
        self.assertEqual(
            "resultado antigo de listar_curriculos compactado; ids citados: cv-1, v-2; chame a tool de novo se precisar do conteudo",
            linha,
        )


if __name__ == "__main__":
    unittest.main()
