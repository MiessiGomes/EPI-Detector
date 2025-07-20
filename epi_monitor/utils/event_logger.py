import datetime
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from epi_monitor.config.settings import SETTINGS

logger = logging.getLogger(__name__)


def log_event(frame_with_detections: np.ndarray, track_id: int) -> Optional[Path]:
    """
    Saves an image of the current frame and returns the path to the saved file.
    """
    log_dir = Path(SETTINGS.EVENT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_ID_{track_id}.jpg"
    filepath = log_dir / filename

    try:
        cv2.imwrite(str(filepath), frame_with_detections)
        logger.info(f"Event image saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Could not save event image to {filepath}. Reason: {e}")
        return None