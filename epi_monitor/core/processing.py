from typing import Any, Dict, List, Tuple

import numpy as np
from ultralytics.engine.results import Boxes

from epi_monitor.config.settings import SETTINGS
from epi_monitor.core.detector import Detector
from epi_monitor.core.evaluator import Evaluator

DetectionDict = Dict[str, Any]


def process_person_track(
    box: Boxes,
    frame: np.ndarray,
    epi_detector: Detector,
    evaluator: Evaluator,
    epi_class_names: Dict[int, str],
    epi_model_person_class_id: int,
) -> Tuple[List[DetectionDict], List[str], int]:
    """
    Processes a single tracked person to detect EPIs and check compliance.
    Returns a list of all detections (person + EPIs), a list of alerts, and the track ID.
    """
    track_id = int(box.id.item())
    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
    detections_to_draw = []

    person_detection: DetectionDict = {
        "name": "Pessoa",
        "box": [x1, y1, x2, y2],
        "confidence": box.conf.item(),
        "class_id": int(box.cls.item()),
        "track_id": track_id,
    }
    detections_to_draw.append(person_detection)

    person_crop: np.ndarray = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return detections_to_draw, [], track_id

    # --- STAGE 2: DETECT EPIs ON THE CROPPED PERSON ---
    epi_results = epi_detector.detect(
        person_crop, conf=SETTINGS.EPI_CONFIDENCE_THRESHOLD
    )
    epi_detections_on_person: List[DetectionDict] = []
    for epi_res in epi_results:
        for epi_box in epi_res.boxes:
            class_id: int = int(epi_box.cls)
            if class_id == epi_model_person_class_id:
                continue

            ex1_rel, ey1_rel, ex2_rel, ey2_rel = [int(i) for i in epi_box.xyxy[0]]
            epi_detection: DetectionDict = {
                "name": epi_class_names[class_id],
                "box": [ex1_rel + x1, ey1_rel + y1, ex2_rel + x1, ey2_rel + y1],
                "confidence": epi_box.conf.item(),
                "class_id": class_id,
            }
            detections_to_draw.append(epi_detection)
            epi_detections_on_person.append(epi_detection)

    alerts = evaluator.check_compliance(epi_detections_on_person)
    return detections_to_draw, alerts, track_id
