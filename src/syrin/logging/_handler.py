"""SyrinHandler — structured logging handler (JSON or text)."""

from __future__ import annotations

import datetime
import json
import logging
import sys
from enum import StrEnum
from typing import TextIO


class LogFormat(StrEnum):
    """Output format for SyrinHandler.

    Attributes:
        JSON: Machine-readable JSON lines (one JSON object per log record). Use in
            production with log aggregators (Datadog, Splunk, CloudWatch, etc.).
        TEXT: Human-readable text. Use in development or when reading logs directly.
    """

    JSON = "json"
    TEXT = "text"


class SyrinHandler(logging.Handler):
    """Structured log handler for Syrin loggers.

    Emits either JSON lines or human-readable text, depending on ``format``.
    Attach to any logger with ``logger.addHandler(SyrinHandler(...))``.

    Args:
        format: ``LogFormat.JSON`` for JSON lines, ``LogFormat.TEXT`` for text. Default: ``JSON``.
        stream: Output stream. Default: ``sys.stdout``.
        stream_override: For testing — if a list is passed, records are appended as strings.

    Example::

        import logging
        from syrin.logging import SyrinHandler, LogFormat

        logging.getLogger("syrin").addHandler(SyrinHandler(format=LogFormat.JSON))
    """

    def __init__(
        self,
        format: LogFormat = LogFormat.JSON,  # noqa: A002
        stream: TextIO | None = None,
        stream_override: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._format = format
        self._stream: TextIO = stream or sys.stdout
        self._stream_override = stream_override

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as JSON or text."""
        try:
            if self._format == LogFormat.JSON:
                line = self._format_json(record)
            else:
                line = self._format_text(record)
            if self._stream_override is not None:
                self._stream_override.append(line)
            else:
                self._stream.write(line + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        data: dict[str, object] = {
            "level": record.levelname,
            "ts": ts,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = logging.Formatter().formatException(record.exc_info)
        # Include extra fields if any
        for key, val in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "taskName",
            ) and not key.startswith("_"):
                data[key] = val
        return json.dumps(data, default=str)

    def _format_text(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC).strftime("%H:%M:%S")
        return f"[{ts}] {record.levelname:<8} {record.name}: {record.getMessage()}"
