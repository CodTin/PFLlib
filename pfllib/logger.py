import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str = "pfllib") -> logging.Logger:
    return logging.getLogger(name)


def setup_logger(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    logger = logging.getLogger("pfllib")
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
