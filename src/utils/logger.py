"""
Centralized logger with consistent formatting.
Logs are saved under logs/ as logs_yyyymmdd_hhmm.log.
"""
import logging
import sys
import os
from datetime import datetime
from typing import Any, Optional


class Logger:
    """
    Logger with a single, readable format. Duck-type compatible with logging.Logger.
    Use Logger.get("name") to obtain a configured logger.
    Prints all log levels to file, info+ to console.
    """

    DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s  %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        format_string: Optional[str] = None,
        stream: Any = None,
    ):
        self._name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)  # Accept all logs, control via handlers
        self._logger.handlers.clear()
        self._logger.propagate = False

        # Ensure logs/ directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        now = datetime.now()
        fname = f"logs_{now.strftime('%Y%m%d_%H%M')}.log"
        log_path = os.path.join(logs_dir, fname)

        formatter = logging.Formatter(format_string or self.DEFAULT_FORMAT, self.DATE_FORMAT)

        # File handler for writing all logs to disk (DEBUG+)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # Stream handler for stderr (console): INFO and higher
        if stream is not None:
            stream_handler = logging.StreamHandler(stream)
        else:
            stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    @classmethod
    def get(
        cls,
        name: str,
        level: int = logging.INFO,
        format_string: Optional[str] = None,
    ) -> "Logger":
        """Return a configured Logger instance for the given name."""
        return cls(name=name, level=level, format_string=format_string)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(msg, *args, **kwargs)

    @property
    def name(self) -> str:
        return self._name
