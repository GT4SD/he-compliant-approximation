"""Utility functions for the logging."""

import logging
from typing import Optional

LOGGING_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logger(
    logger_name: str, logging_level: str = "info", logging_file: Optional[str] = None
) -> logging.Logger:
    """Configures the logging for the application.

    Args:
        name (str): The name of the logger.
        logging_level (str): The logging level. Defaults to "info".
        logging_file (Optional[str]): The file to log to. Defaults to None.

    Returns:
        logging.Logger: The configured logger.

    Raises:
        ValueError: If the logging level is invalid.
    """
    logging_level = logging_level.lower()
    if logging_level not in LOGGING_LEVELS:
        raise ValueError(
            f"Invalid logging level: {logging_level}. Available levels: {LOGGING_LEVELS.keys()}"
        )
    LOGGER = logging.getLogger(logger_name)
    logging.basicConfig(
        level=LOGGING_LEVELS[logging_level],
        format="%(name)s - %(levelname)s - %(message)s",
        filename=logging_file,
    )
    return LOGGER
