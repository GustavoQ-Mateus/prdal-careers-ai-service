from datetime import datetime
from typing import Any

TIPOS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _data_iso(valor: str) -> bool:
    try:
        datetime.fromisoformat(valor.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def erros(valor: Any, esquema: dict[str, Any], caminho: str = "$") -> list[str]:
    achados: list[str] = []
    tipo = esquema.get("type")
    if tipo is not None:
        tipos = tipo if isinstance(tipo, list) else [tipo]
        if not any(TIPOS[t](valor) for t in tipos):
            return [f"{caminho}: esperado {'/'.join(tipos)}"]
    if "enum" in esquema and valor not in esquema["enum"]:
        achados.append(f"{caminho}: fora de {esquema['enum']}")
    if esquema.get("format") == "date-time" and isinstance(valor, str) and not _data_iso(valor):
        achados.append(f"{caminho}: data invalida")
    if isinstance(valor, dict):
        propriedades = esquema.get("properties", {})
        for nome in esquema.get("required", []):
            if nome not in valor:
                achados.append(f"{caminho}.{nome}: obrigatorio")
        for nome, item in valor.items():
            if nome in propriedades:
                achados += erros(item, propriedades[nome], f"{caminho}.{nome}")
            elif esquema.get("additionalProperties") is False:
                achados.append(f"{caminho}.{nome}: campo nao aceito")
    if isinstance(valor, list) and isinstance(esquema.get("items"), dict):
        for indice, item in enumerate(valor):
            achados += erros(item, esquema["items"], f"{caminho}[{indice}]")
    return achados
