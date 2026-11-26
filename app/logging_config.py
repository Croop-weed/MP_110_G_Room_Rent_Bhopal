"""JSON application logging and request-ID middleware support."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class JsonFormatter(logging.Formatter):
    """Serialize application records to one structured JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure stdout JSON logging once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Add a request ID to request state, response, and application logs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        logger = logging.getLogger("pg_rent.application")
        logger.info(
            "Incoming request",
            extra={"request_id": request_id},
        )
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "Request completed",
            extra={"request_id": request_id},
        )
        return response
