from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel

raw_env = dotenv_values(".env")


class Settings(BaseModel):
    """
    Validates and holds all project settings using Pydantic.
    """

    # --- Discord Webhook ---
    DISCORD_WEBHOOK_URL: str

    # --- Model Configurations ---
    MODEL_PERSON_PATH: Path
    EPI_DETECTOR_PATH: Path
    CONFIDENCE_THRESHOLD: float = 0.5

    # --- Video Processing ---
    VIDEO_SOURCE: str = "0"
    SKIP_FRAMES: int = 2

    # --- Training Parameters ---
    DATA_PATH: Path
    EPOCHS: int = 100
    IMG_SIZE: int = 640
    BATCH_SIZE: int = 16


SETTINGS = Settings.model_validate(raw_env)
