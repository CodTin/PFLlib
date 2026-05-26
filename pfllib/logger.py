import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_FMT = "[{time:YYYY-MM-DD HH:mm:ss}][{extra[name]}][{level}] {message}"

logger.remove()
logger.add(sys.stderr, format=_FMT, level="INFO")


def get_logger(name: str = "pfllib"):
    return logger.bind(name=name)


def setup_logger(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    import logging

    logging.getLogger("pfllib").setLevel(getattr(logging, level.upper(), logging.INFO))

    logger.remove()
    logger.add(sys.stderr, format=_FMT, level=level.upper())

    if log_file:
        logger.add(str(log_file), format=_FMT, level=level.upper())