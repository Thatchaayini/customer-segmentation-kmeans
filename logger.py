import logging
import os
import time
from functools import wraps

from config import LOG_FILE, LOG_LEVEL


def get_logger(name):
    """Return a logger that writes to the console and to logs/segmentation.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def timed_step(func):
    """Decorator: log when a step starts, finishes (with seconds) or fails."""
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("Started: %s", func.__name__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.exception("Failed: %s", func.__name__)
            raise
        logger.info("Finished: %s in %.2fs", func.__name__, time.perf_counter() - start)
        return result

    return wrapper