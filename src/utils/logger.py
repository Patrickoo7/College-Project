"""Logging module for the heart disease prediction system."""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

from .config import get_config


class Logger:
    """Custom logger with file and console handlers."""

    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: Optional[str] = None,
    ):
        """
        Initialize logger.

        Args:
            name: Logger name (usually __name__ of the calling module)
            log_file: Path to log file. If None, uses config default
            level: Logging level. If None, uses config default
        """
        self.logger = logging.getLogger(name)

        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            return

        config = get_config()

        # Set logging level
        if level is None:
            level = config.get("logging.level", "INFO")
        self.logger.setLevel(getattr(logging, level))

        # Set format
        log_format = config.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file is None:
            log_file = config.get("logging.file", "logs/app.log")

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = config.get("logging.max_bytes", 10485760)  # 10MB
        backup_count = config.get("logging.backup_count", 5)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)


def get_logger(name: str, log_file: Optional[str] = None, level: Optional[str] = None) -> Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
        level: Optional logging level

    Returns:
        Logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        >>> logger.error("This is an error message")
    """
    return Logger(name, log_file, level)


# Convenience function for quick logging
def setup_logging(level: str = "INFO") -> None:
    """
    Setup basic logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
