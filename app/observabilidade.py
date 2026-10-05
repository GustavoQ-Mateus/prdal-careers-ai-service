import contextvars
import json
import logging
import re
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from starlette.requests import Request

HEADER_REQUEST_ID = "X-Request-Id"
SERVICO = "ai-service"
ROTAS_SILENCIOSAS = frozenset({"/health", "/ready"})
MENSAGEM_DESLIGAMENTO = "O servico de IA esta reiniciando e a resposta foi interrompida. Envie a mensagem de novo para continuar."

_REQUEST_ID_VALIDO = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_CAMPOS_PADRAO = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

desligando = threading.Event()
acesso = logging.getLogger("prdal.http")


def request_id_atual() -> str | None:
    return _request_id.get()


def request_id_de(recebido: str | None) -> str:
    return recebido if recebido and _REQUEST_ID_VALIDO.match(recebido) else str(uuid.uuid4())


class FormatadorJson(logging.Formatter):
    def format(self, registro: logging.LogRecord) -> str:
        linha: dict[str, object] = {
            "horario": datetime.fromtimestamp(registro.created, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "nivel": registro.levelname.lower(),
            "servico": SERVICO,
            "contexto": registro.name,
        }
        request_id = getattr(registro, "requestId", None) or request_id_atual()
        if request_id:
            linha["requestId"] = request_id
        linha["mensagem"] = registro.getMessage()
        for chave, valor in vars(registro).items():
            if chave not in _CAMPOS_PADRAO and chave not in linha:
                linha[chave] = valor
        if registro.exc_info:
            linha["stack"] = self.formatException(registro.exc_info)
        return json.dumps(linha, ensure_ascii=False, default=str)


def configurar_logs(nivel: int = logging.INFO) -> None:
    saida = logging.StreamHandler(sys.stdout)
    saida.setFormatter(FormatadorJson())
    raiz = logging.getLogger()
    raiz.handlers = [saida]
    raiz.setLevel(nivel)
    for nome in ("uvicorn", "uvicorn.error"):
        logging.getLogger(nome).handlers = []
        logging.getLogger(nome).propagate = True
    logging.getLogger("uvicorn.access").disabled = True


async def middleware_requisicao(request: Request, call_next):
    request_id = request_id_de(request.headers.get(HEADER_REQUEST_ID))
    marca = _request_id.set(request_id)
    inicio = time.perf_counter()
    status = 500
    try:
        resposta = await call_next(request)
        status = resposta.status_code
        resposta.headers[HEADER_REQUEST_ID] = request_id
        return resposta
    finally:
        if request.url.path not in ROTAS_SILENCIOSAS:
            acesso.info(
                "requisicao",
                extra={
                    "requestId": request_id,
                    "metodo": request.method,
                    "rota": request.url.path,
                    "status": status,
                    "duracaoMs": round((time.perf_counter() - inicio) * 1000),
                },
            )
        _request_id.reset(marca)


def avisar_desligamento() -> None:
    for sinal in (signal.SIGTERM, signal.SIGINT):
        anterior = signal.getsignal(sinal)

        def tratar(numero, quadro, anterior=anterior):
            if not desligando.is_set():
                logging.getLogger("prdal.desligamento").info("desligamento iniciado", extra={"sinal": signal.Signals(numero).name})
            desligando.set()
            if callable(anterior):
                anterior(numero, quadro)

        try:
            signal.signal(sinal, tratar)
        except ValueError:
            return
