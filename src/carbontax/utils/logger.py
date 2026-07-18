"""
Logging setup: one timestamped file under logs/ (DEBUG+) plus stderr (INFO+).

Entry-point ``main()`` functions call ``setup_logging()`` once; modules obtain
their logger with the standard ``logging.getLogger(__name__)``.
"""
import logging
import os
import sys
from datetime import datetime

DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGS_DIR = "logs"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once; repeated calls are no-ops."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(DEFAULT_FORMAT, DATE_FORMAT)

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"logs_{datetime.now():%Y%m%d_%H%M}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
