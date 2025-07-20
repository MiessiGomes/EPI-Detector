from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel

env_path = Path(__file__).parent.parent.parent / ".env"

raw_env = dotenv_values(dotenv_path=env_path)


class Settings(BaseModel):
    """
    Validates and holds all project settings using Pydantic.
    """

    # --- Infrastructure ---
    DISCORD_WEBHOOK_URL: str
    DATABASE_URL: str  # Example: "postgresql://user:password@host:port/dbname"
    CAMERA_ID: str = "CAM-01"

    # --- Model Configurations ---
    MODEL_PERSON_PATH: Path
    EPI_DETECTOR_PATH: Path
    PERSON_CLASS_NAME: str = "person"
    EPI_MODEL_IGNORE_CLASS_NAME: str = "Pessoa"
    EPI_CONFIDENCE_THRESHOLD: float = 0.65
    PERSON_CONFIDENCE_THRESHOLD: float = 0.65

    # --- Video Processing ---
    VIDEO_SOURCE: str = "0"
    SKIP_FRAMES: int = 2
    RECORD_VIDEO: bool = False
    OUTPUT_VIDEO_PATH: Path

    # --- Event Logging ---
    EVENT_LOG_DIR: Path = "events"
    LOG_FILE_PATH: Path = "epi_monitor/logs/app.log"

    # --- Training Parameters ---
    DATA_PATH: Path
    EPOCHS: int = 100
    IMG_SIZE: int = 640
    BATCH_SIZE: int = 16


SETTINGS = Settings.model_validate(raw_env)
