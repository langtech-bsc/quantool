
import sys
import threading
import json

from loguru import logger

_logger = logger

class LoggerFactory:
    _lock = threading.Lock()
    _configured = False

    @classmethod
    def configure(cls, level="INFO", fmt=None, json_format=False):
        """Configure loguru logger once with optional JSON formatting."""
        with cls._lock:
            if cls._configured:
                return

            # Remove default handler
            _logger.remove()

            if json_format:
                def _json_patcher(record):
                    original_message = record.get("message", "")
                    serialized = {
                        "timestamp": record["time"].isoformat(),
                        "level": record["level"].name,
                        "name": record["name"],
                        "message": original_message,
                        "module": record["module"],
                        "function": record["function"],
                        "line": record["line"]
                    }

                    if record.get("extra"):
                        try:
                            serialized["extra"] = json.loads(json.dumps(record["extra"], default=str))
                        except (TypeError, ValueError):
                            serialized["extra"] = {key: str(value) for key, value in record["extra"].items()}

                    exception = record.get("exception")
                    if exception:
                        serialized["exception"] = {
                            "type": exception.type.__name__ if exception.type else None,
                            "value": str(exception.value) if exception.value else None
                        }

                    record["message"] = json.dumps(serialized)

                _logger.configure(patcher=_json_patcher)

                _logger.add(
                    sys.stderr,
                    format="{message}",
                    level=level,
                    backtrace=True,
                    diagnose=True
                )

                _logger.add(
                    "logs/quantool.log.json",
                    format="{message}",
                    rotation="10 MB",
                    retention="10 days",
                    level=level,
                    backtrace=True,
                    diagnose=True
                )
            else:
                # Include function name and line number in log output
                fmt = fmt or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <8}</level> <magenta>[{name}:{module}:{function}:{line}]</magenta> <level>{message}</level>"
                _logger.add(
                    sys.stderr,
                    format=fmt,
                    level=level,
                    backtrace=True,
                    colorize=True,
                    diagnose=True
                )
            
                _logger.add(
                    "logs/quantool.log",
                    rotation="10 MB",
                    retention="10 days",
                    level=level,
                    backtrace=True,
                    diagnose=True
                )

            cls._configured = True

    @classmethod
    def get_logger(cls, name=None, **kwargs):
        """Get a module-level logger after ensuring configuration."""
        cls.configure(**kwargs)
        if name:
            return _logger.bind(name=name)
        return _logger

    @classmethod
    def configure_external_logger(cls, external_logger, level="INFO", fmt=None, json_format=False):
        """
        Configure an external logger (like llmcompressor's logger) to use quantool's format.
        
        Parameters
        ----------
        external_logger : loguru.Logger
            The external logger instance to configure
        level : str
            Logging level (default: "INFO")
        fmt : str, optional
            Custom format string for the logger
        json_format : bool
            Whether to use JSON formatting (default: False)
            
        Example
        -------
        >>> from llmcompressor import logger as llmcompressor_logger
        >>> LoggerFactory.configure_external_logger(llmcompressor_logger)
        """
        # Remove existing handlers from the external logger
        external_logger.remove()
        
        if json_format:
            def _json_patcher(record):
                original_message = record.get("message", "")
                serialized = {
                    "timestamp": record["time"].isoformat(),
                    "level": record["level"].name,
                    "name": record["name"],
                    "message": original_message,
                    "module": record["module"],
                    "function": record["function"],
                    "line": record["line"]
                }

                if record.get("extra"):
                    try:
                        serialized["extra"] = json.loads(json.dumps(record["extra"], default=str))
                    except (TypeError, ValueError):
                        serialized["extra"] = {key: str(value) for key, value in record["extra"].items()}

                exception = record.get("exception")
                if exception:
                    serialized["exception"] = {
                        "type": exception.type.__name__ if exception.type else None,
                        "value": str(exception.value) if exception.value else None
                    }

                record["message"] = json.dumps(serialized)

            external_logger.configure(patcher=_json_patcher)

            external_logger.add(
                sys.stderr,
                format="{message}",
                level=level,
                backtrace=True,
                diagnose=True
            )
            external_logger.add(
                "logs/quantool.log.json",
                format="{message}",
                rotation="10 MB",
                retention="10 days",
                level=level,
                backtrace=True,
                diagnose=True
            )
        else:
            # Include function name and line number in log output
            fmt = fmt or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <8}</level> <magenta>[{name}:{module}:{function}:{line}]</magenta> <level>{message}</level>"
            external_logger.add(
                sys.stderr,
                format=fmt,
                level=level,
                backtrace=True,
                colorize=True,
                diagnose=True
            )
            external_logger.add(
                "logs/quantool.log",
                rotation="10 MB",
                retention="10 days",
                level=level,
                backtrace=True,
                diagnose=True
            )