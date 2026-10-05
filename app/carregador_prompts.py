import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

DIRETORIO_PROMPTS = Path(__file__).resolve().parent / "prompts"
PROMPTS_OBRIGATORIOS = ("reescrita", "juiz_relacao")

_ARQUIVO_RE = re.compile(r"^(?P<id>[a-z0-9_-]+)\.v(?P<versao>\d+)\.md$")
_CABECALHO_RE = re.compile(r"\A---\n(?P<campos>.*?)\n---\n(?P<texto>.*)\Z", re.DOTALL)


class PromptAusente(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    id: str
    versao: int
    texto: str
    sha256: str

    @property
    def rotulo(self) -> str:
        return f"{self.id}.v{self.versao}"


_registro: dict[str, Prompt] = {}


def _ler(arquivo: Path) -> Prompt:
    nome = _ARQUIVO_RE.match(arquivo.name)
    if not nome:
        raise PromptAusente(f"nome de prompt invalido: {arquivo.name}")
    bruto = arquivo.read_text(encoding="utf-8").replace("\r\n", "\n")
    cabecalho = _CABECALHO_RE.match(bruto)
    if not cabecalho:
        raise PromptAusente(f"prompt sem cabecalho de id e versao: {arquivo.name}")
    campos = dict(
        (chave.strip(), valor.strip())
        for chave, _, valor in (linha.partition(":") for linha in cabecalho.group("campos").splitlines())
        if chave.strip()
    )
    if campos.get("id") != nome.group("id") or campos.get("versao") != nome.group("versao"):
        raise PromptAusente(f"id ou versao do cabecalho divergem do nome do arquivo: {arquivo.name}")
    texto = cabecalho.group("texto").strip()
    if not texto:
        raise PromptAusente(f"prompt vazio: {arquivo.name}")
    return Prompt(
        id=nome.group("id"),
        versao=int(nome.group("versao")),
        texto=texto,
        sha256=hashlib.sha256(texto.encode("utf-8")).hexdigest(),
    )


def carregar_prompts(diretorio: Path | None = None) -> dict[str, Prompt]:
    pasta = diretorio or DIRETORIO_PROMPTS
    encontrados: dict[str, Prompt] = {}
    if pasta.is_dir():
        for arquivo in sorted(pasta.glob("*.md")):
            prompt = _ler(arquivo)
            atual = encontrados.get(prompt.id)
            if atual is None or prompt.versao > atual.versao:
                encontrados[prompt.id] = prompt
    faltando = [pid for pid in PROMPTS_OBRIGATORIOS if pid not in encontrados]
    if faltando:
        raise PromptAusente(f"prompt obrigatorio ausente em {pasta}: {', '.join(faltando)}")
    _registro.clear()
    _registro.update(encontrados)
    return dict(encontrados)


def obter(prompt_id: str) -> Prompt:
    if prompt_id not in _registro:
        carregar_prompts()
    return _registro[prompt_id]
