import json
import logging
import os
from typing import Any

from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger("prdal.llm")

_provedor: Any = None
_tracer: Any = None


def _logger_visivel() -> None:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        manipulador = logging.StreamHandler()
        manipulador.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(manipulador)
        logger.propagate = False


def configurar(provedor: Any = None) -> None:
    global _provedor, _tracer
    if provedor is None and os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip():
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provedor = TracerProvider(
            resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "ai-service")})
        )
        provedor.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    _provedor = provedor
    _tracer = provedor.get_tracer("prdal.ai-service") if provedor is not None else None
    if _tracer is None:
        _logger_visivel()


def encerrar() -> None:
    if _provedor is not None and hasattr(_provedor, "shutdown"):
        _provedor.shutdown()


def exportando() -> bool:
    return _tracer is not None


def registrar_chamada(atributos: dict[str, Any], inicio_ns: int, fim_ns: int, erro: str | None = None) -> None:
    limpos = {chave: valor for chave, valor in atributos.items() if valor is not None}
    if _tracer is None:
        logger.info(json.dumps({"evento": "chamada_llm", **limpos}, ensure_ascii=False, sort_keys=True, default=list))
        return
    span = _tracer.start_span(
        f"{limpos.get('gen_ai.operation.name', 'chat')} {limpos.get('gen_ai.request.model', '')}".strip(),
        kind=SpanKind.CLIENT,
        start_time=inicio_ns,
        attributes=limpos,
    )
    if erro:
        span.set_status(Status(StatusCode.ERROR, erro))
    span.end(end_time=fim_ns)
