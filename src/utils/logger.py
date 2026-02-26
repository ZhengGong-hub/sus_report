"""
Centralized logger with consistent formatting.
"""
import logging
import sys
from typing import Any, Optional


class Logger:
    """
    Logger with a single, readable format. Duck-type compatible with logging.Logger.
    Use Logger.get("name") to obtain a configured logger.
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
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        self._logger.propagate = False

        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(format_string or self.DEFAULT_FORMAT, self.DATE_FORMAT)
        )
        self._logger.addHandler(handler)

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
