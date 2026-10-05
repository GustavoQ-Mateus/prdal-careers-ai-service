import io
import re
import subprocess
import tokenize
from pathlib import Path

raiz = Path(__file__).resolve().parents[1]
arquivos = subprocess.check_output(
    ["git", "ls-files", "-z", "--", "."], cwd=raiz
).decode("utf-8").split("\0")
falhas = 0


def acusar(arquivo, linha, motivo):
    global falhas
    print(f"{arquivo}:{linha}: {motivo}")
    falhas += 1


for arquivo in filter(None, arquivos):
    caminho = raiz / arquivo
    if not caminho.is_file():
        continue
    dados = caminho.read_bytes()
    texto = dados.decode("utf-8", errors="replace")
    posicao = texto.find(chr(0x2014))
    if posicao >= 0:
        acusar(arquivo, texto[:posicao].count("\n") + 1, "travessao proibido")
    if caminho.suffix == ".py":
        try:
            for token in tokenize.tokenize(io.BytesIO(dados).readline):
                if token.type != tokenize.COMMENT:
                    continue
                linha = token.start[0]
                if linha == 1 and token.string.startswith("#!"):
                    continue
                if linha <= 2 and re.match(r"#.*?coding[:=]\s*[-\w.]+", token.string):
                    continue
                acusar(arquivo, linha, "comentario em codigo")
        except (tokenize.TokenError, SyntaxError) as erro:
            acusar(arquivo, 1, f"Python invalido: {erro}")

if falhas:
    raise SystemExit(1)
print(f"Texto verificado: {sum(bool(a) for a in arquivos)} arquivos")
