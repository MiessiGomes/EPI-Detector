from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: Path):
        """
        Initializes the detector with a YOLO model.
        """
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, conf: float) -> Any:
        """
        Performs object detection on a single frame with a given confidence threshold.
        """
        return self.model(frame, conf=conf, device=0)
