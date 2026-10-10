"""Structured JSON logging. Log pseudonymised text only; never raw queries/prompts containing PII."""
from __future__ import annotations

import json
import logging
import os
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "ragkit") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
