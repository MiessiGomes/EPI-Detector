import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from epi_monitor.config.settings import SETTINGS


def setup_logging():
    """
    Configures the root logger for the application.
    Logs are sent to both the console and a time-rotating file.
    """
    log_filepath = Path(SETTINGS.LOG_FILE_PATH)

    log_filepath.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- Time-Rotating File Handler ---
    # Rotates the log file every 24 hours at midnight
    file_handler = TimedRotatingFileHandler(
        filename=log_filepath,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
