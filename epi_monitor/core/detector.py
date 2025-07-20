from pathlib import Path
from typing import Any

import numpy as np
from config.settings import SETTINGS
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> Any:
        """
        Performs object detection on a single frame.
        """
        return self.model(frame, conf=SETTINGS.CONFIDENCE_THRESHOLD, device=0)
