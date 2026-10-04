import json
import unittest

from app.copiloto import planejar_turno, redigir_formulario, redigir_mensagem
from app.generate import generate_cv_pipeline
from app.keywords import extract_keywords
from app.schemas import (
    GenerateCvRequest,
    RedigirFormularioRequest,
    RedigirMensagemRequest,
    TurnRequest,
)
from tests.cliente_falso import ComClienteFalso, resposta, resposta_blocos
from tests.perfis import frase, reescrita, reparo


def _perfil(nome: str, empresa: str, tecnologia: str) -> dict:
    return {
        "nome": nome,
        "emails": [{"valor": f"{nome.split()[0].lower()}@exemplo.dev", "principal": True}],
        "resumo": f"Pessoa desenvolvedora com foco em {tecnologia}.",
        "experiencias": [
            {
                "cargo": "Desenvolvedora",
                "empresa": empresa,
                "dataInicioMes": 1,
                "dataInicioAno": 2022,
                "atual": True,
                "descricao": f"Tecnologias: {tecnologia}",
                "realizacoes": [f"Desenvolvi servicos em {tecnologia} para pagamentos."],
            }
        ],
        "skills": [tecnologia],
    }


def _geracao(nome, empresa, tecnologia, vaga):
    return GenerateCvRequest.model_validate(
        {
            "perfilMestre": _perfil(nome, empresa, tecnologia),
            "vaga": {"titulo": vaga, "empresa": f"Empresa {vaga}", "descricao": f"Vaga de {vaga} com {tecnologia}."},
            "keywords": [{"termo": tecnologia, "peso": 1}],
            "contexto": [f"Projeto interno com {tecnologia}."],
        }
    )


def _turno(texto: str, oportunidade: str | None) -> TurnRequest:
    return TurnRequest.model_validate(
        {
            "oportunidadeId": oportunidade,
            "mensagens": [{"role": "user", "content": [{"type": "text", "text": texto}]}],
            "tools": [{"name": "ler_perfil", "description": "Le o perfil", "input_schema": {"type": "object", "properties": {}}}],
        }
    )


class PrefixoEmCacheTest(unittest.TestCase):
    def assertPrefixoEstavel(self, requisicoes, variaveis):
        self.assertGreaterEqual(len(requisicoes), 2)
        primeira = requisicoes[0]
        for outra in requisicoes[1:]:
            self.assertEqual(json.dumps(primeira["system"]), json.dumps(outra["system"]))
            self.assertEqual(json.dumps(primeira["output_config"]), json.dumps(outra["output_config"]))
        for requisicao in requisicoes:
            ultimo = requisicao["system"][-1]
            self.assertEqual({"type": "ephemeral"}, ultimo["cache_control"])
            self.assertEqual(1, json.dumps(requisicao).count("cache_control"))
            prefixo = json.dumps(requisicao["system"], ensure_ascii=False)
            for variavel in variaveis:
                self.assertNotIn(variavel, prefixo)
        self.assertNotEqual(
            json.dumps(requisicoes[0]["messages"]), json.dumps(requisicoes[-1]["messages"])
        )

    def test_keywords(self):
        ok = {"keywords": [{"termo": "Python", "peso": 1, "tipo": "stack"}]}
        with ComClienteFalso(resposta(ok), resposta(ok)) as cliente:
            extract_keywords("Vaga de backend com Python e Django.")
            extract_keywords("Vaga de dados com Spark e Airflow.")
        self.assertPrefixoEstavel(cliente.requisicoes, ["Django", "Airflow"])

    def test_turno_do_copiloto(self):
        texto = [{"type": "text", "text": "Certo."}]
        with ComClienteFalso(resposta_blocos(texto), resposta_blocos(texto)) as cliente:
            planejar_turno(_turno("quero ver minhas vagas", None))
            planejar_turno(_turno("gere o curriculo da vaga", "op-123"))
        for requisicao in cliente.requisicoes:
            self.assertEqual({"type": "ephemeral"}, requisicao["system"][-1]["cache_control"])
            self.assertEqual({"type": "ephemeral"}, requisicao["messages"][-1]["content"][-1]["cache_control"])
            self.assertEqual(2, json.dumps(requisicao).count("cache_control"))
        primeira, segunda = cliente.requisicoes
        for chave in ("system", "tools", "tool_choice", "output_config"):
            self.assertEqual(json.dumps(primeira[chave]), json.dumps(segunda[chave]))
        prefixo = json.dumps([primeira["tools"], primeira["system"]], ensure_ascii=False)
        for variavel in ("minhas vagas", "op-123"):
            self.assertNotIn(variavel, prefixo)

    def test_redacoes(self):
        mensagem = {"titulo": "t", "texto": "Ola", "destino": "email"}
        with ComClienteFalso(resposta(mensagem), resposta(mensagem)) as cliente:
            redigir_mensagem(RedigirMensagemRequest.model_validate({"vaga": {"titulo": "Backend"}, "contexto": "indicacao"}))
            redigir_mensagem(RedigirMensagemRequest.model_validate({"vaga": {"titulo": "Dados"}, "contexto": "evento"}))
        self.assertPrefixoEstavel(cliente.requisicoes, ["Backend", "Dados", "indicacao", "evento"])

        formulario = {"titulo": "t", "respostas": [{"campo": "c", "texto": "r"}], "texto": "r"}
        with ComClienteFalso(resposta(formulario), resposta(formulario)) as cliente:
            redigir_formulario(RedigirFormularioRequest.model_validate({"vaga": {"titulo": "Backend"}, "campos": ["Pretensao"]}))
            redigir_formulario(RedigirFormularioRequest.model_validate({"vaga": {"titulo": "Dados"}, "campos": ["Disponibilidade"]}))
        self.assertPrefixoEstavel(cliente.requisicoes, ["Pretensao", "Disponibilidade"])

    def test_reescrita_entre_geracoes_e_reparo(self):
        com_rejeicao = reescrita(
            frase("Desenvolvedora", "skills"),
            [frase("Projeto interno de destaque.", "contexto-1")],
            [],
        )
        respostas = [resposta(com_rejeicao), resposta(reparo())] * 2
        with ComClienteFalso(*respostas) as cliente:
            generate_cv_pipeline(_geracao("Ana Souza", "Banco Alfa", "Python", "Backend Pleno"))
            generate_cv_pipeline(_geracao("Bruno Lima", "Loja Beta", "Kotlin", "Mobile Senior"))
        self.assertEqual(4, len(cliente.requisicoes))
        self.assertPrefixoEstavel(
            cliente.requisicoes,
            ["Ana Souza", "Bruno Lima", "Banco Alfa", "Loja Beta", "Kotlin", "Backend Pleno", "Mobile Senior"],
        )


if __name__ == "__main__":
    unittest.main()
