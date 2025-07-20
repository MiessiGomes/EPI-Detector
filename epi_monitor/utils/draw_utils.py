from typing import Any, Dict, List

import cv2
import numpy as np

# A dictionary mapping class names to specific BGR colors.
CLASS_NAME_COLORS: Dict[str, tuple[int, int, int]] = {
    # --- EPI Classes ---
    "Abafador de ruido": (220, 20, 60),  # Crimson
    "Abafador de ruido com": (119, 11, 32),  # Dark Red
    "Botas de seguranca": (255, 255, 0),  # Navy
    "Capacete de seguranca": (0, 0, 230),  # Blue
    "Capacete de seguranca com": (106, 90, 205),  # Slate Blue
    "Luvas de protecao": (0, 60, 100),  # Dark Cyan
    "Luvas de protecao com": (0, 80, 100),  # Darker Cyan
    "Mascara": (0, 255, 255),  # Aqua
    "Oculos de protecao": (0, 100, 0),  # Dark Green
    "Oculos de protecao com": (0, 255, 0),  # Lime
    "Roupa de protecao": (255, 0, 0),  # Red
    "Pessoa com": (255, 255, 0),  # Yellow
    # --- Person Class (from the primary detector) ---
    "Pessoa": (220, 220, 220),  # Light Gray
    # Default color for any other class
    "default": (0, 255, 0),  # Green
}

DetectionDict = Dict[str, Any]


def draw_detections(frame: np.ndarray, detections: List[DetectionDict]) -> np.ndarray:
    """
    Draws bounding boxes and labels for all detections on the frame.
    """
    for detection in detections:
        box = [int(i) for i in detection["box"]]
        x1, y1, x2, y2 = box

        track_id = detection.get("track_id")
        id_text = f"ID: {track_id} | " if track_id is not None else ""
        label = f"{id_text}{detection['name']}: {detection['confidence']:.2f}"

        color = CLASS_NAME_COLORS.get(detection["name"], CLASS_NAME_COLORS["default"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3
        )
    return frame


def draw_compliance_status(
    frame: np.ndarray, non_compliant_alerts: List[str]
) -> np.ndarray:
    """
    Draws compliance status alerts on the frame.
    """
    if non_compliant_alerts:
        for i, alert in enumerate(non_compliant_alerts):
            cv2.putText(
                frame,
                alert,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
    else:
        cv2.putText(
            frame,
            "All personnel compliant",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return frame


def draw_final_results(
    frame: np.ndarray, detections: List[DetectionDict], alerts: List[str]
) -> np.ndarray:
    """Draws all detections and compliance alerts on the frame."""
    frame = draw_detections(frame, detections)
    frame = draw_compliance_status(frame, alerts)
    return cv2.resize(frame, (800, 600))
