from app.keywords import interpretar_lote, preparar_lote
from app.main import InterpretarLoteRequest, PrepararLoteRequest, ResultadoLote, VagaLote, interpretar_keywords_lote, preparar_keywords_lote


def test_preparar_reutiliza_prompt_e_schema(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "modelo-teste")
    pedido = preparar_lote("Python e SQL")
    assert pedido["model"] == "modelo-teste"
    assert "Python e SQL" in pedido["messages"][0]["content"][0]["text"]
    assert pedido["output_config"]["format"]["type"] == "json_schema"


def test_interpretar_valida_keywords_e_isola_falha():
    resposta = {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"keywords":[{"termo":"Python","peso":0.8,"tipo":"stack"}]}'}]}
    assert interpretar_lote(resposta)[0].termo == "Python"
    itens = interpretar_keywords_lote(InterpretarLoteRequest(resultados=[
        ResultadoLote(id="ok", resposta=resposta),
        ResultadoLote(id="erro", erro="provedor indisponivel"),
    ]))["itens"]
    assert [item["status"] for item in itens] == ["VALIDAS", "PENDENTE"]


def test_preparacao_isola_falha_de_uma_vaga(monkeypatch):
    def preparar(descricao):
        if descricao == "falha":
            raise ValueError("descricao invalida")
        return {"model": "teste"}

    monkeypatch.setattr("app.main.preparar_lote", preparar)
    resposta = preparar_keywords_lote(PrepararLoteRequest(vagas=[
        VagaLote(id="ok", descricao="Python"),
        VagaLote(id="erro", descricao="falha"),
    ]))
    assert [item["id"] for item in resposta["pedidos"]] == ["ok"]
    assert [item["id"] for item in resposta["erros"]] == ["erro"]
