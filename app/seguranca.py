import hmac
import os

from starlette.requests import Request
from starlette.responses import JSONResponse

HEADER_SERVICO = "X-Prdal-Servico"
TAMANHO_MINIMO_TOKEN = 32
ROTAS_PUBLICAS = frozenset({"/health", "/ready"})


class TokenServicoAusente(RuntimeError):
    pass


def em_desenvolvimento() -> bool:
    return os.getenv("PRDAL_AMBIENTE", "").strip().lower() == "desenvolvimento"


def token_servico() -> str:
    return os.getenv("SERVICE_TOKEN", "").strip()


def exigir_token_no_boot() -> None:
    if em_desenvolvimento():
        return
    if len(token_servico().encode("utf-8")) < TAMANHO_MINIMO_TOKEN:
        raise TokenServicoAusente(
            f"SERVICE_TOKEN ausente ou com menos de {TAMANHO_MINIMO_TOKEN} bytes; "
            "defina o token de servico antes de subir o ai-service fora de PRDAL_AMBIENTE=desenvolvimento"
        )


def rotas_de_documentacao() -> dict[str, str | None]:
    if em_desenvolvimento():
        return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}
    return {"docs_url": None, "redoc_url": None, "openapi_url": None}


def credencial_valida(recebido: str | None) -> bool:
    esperado = token_servico()
    if not esperado:
        return em_desenvolvimento()
    return hmac.compare_digest((recebido or "").encode("utf-8"), esperado.encode("utf-8"))


async def exigir_servico(request: Request, call_next):
    if request.url.path in ROTAS_PUBLICAS or credencial_valida(request.headers.get(HEADER_SERVICO)):
        return await call_next(request)
    return JSONResponse(status_code=401, content={"detail": "credencial de servico invalida"})
