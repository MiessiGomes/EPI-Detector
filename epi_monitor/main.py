import time
from typing import Any, Dict, List

import cv2
import numpy as np
from config.settings import SETTINGS
from core.detector import Detector
from core.evaluator import Evaluator
from core.notifier import Notifier
from utils.draw_utils import draw_compliance_status, draw_detections

DetectionDict = Dict[str, Any]


def main() -> None:
    # --- Stage 1: Person Detector ---
    person_detector = Detector(SETTINGS.MODEL_PERSON_PATH)
    person_class_id: int = list(person_detector.model.names.keys())[
        list(person_detector.model.names.values()).index("person")
    ]

    # --- Stage 2: EPI Detector ---
    epi_detector = Detector(SETTINGS.EPI_DETECTOR_PATH)
    epi_class_names: Dict[int, str] = epi_detector.model.names
    epi_model_person_class_id: int = None
    for cid, name in epi_class_names.items():
        if name == "Pessoa":
            epi_model_person_class_id = cid
            break

    required_ppe_names: List[str] = [
        "Capacete de seguranca",
        "Oculos de protecao",
        "Luvas de protecao",
    ]
    required_ppe_ids: List[int] = [
        cid for cid, name in epi_class_names.items() if name in required_ppe_names
    ]
    evaluator = Evaluator(
        required_ppe_ids=required_ppe_ids, class_names=epi_class_names
    )

    notifier = Notifier(SETTINGS.DISCORD_WEBHOOK_URL)
    cap = cv2.VideoCapture(
        str(SETTINGS.VIDEO_SOURCE)
        if not str(SETTINGS.VIDEO_SOURCE).isdigit()
        else int(SETTINGS.VIDEO_SOURCE)
    )

    if not cap.isOpened():
        print(f"Error: Could not open video source: {SETTINGS.VIDEO_SOURCE}")
        return

    print("Starting 2-Stage EPI Monitor...")
    last_notification_time: float = 0.0
    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break

        person_results = person_detector.detect(frame)

        all_detections_to_draw: List[DetectionDict] = []
        all_alerts: List[str] = []

        for res in person_results:
            for box in res.boxes:
                if int(box.cls) != person_class_id:
                    continue

                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                person_detection: DetectionDict = {
                    "name": "Pessoa",
                    "box": [x1, y1, x2, y2],
                    "confidence": box.conf[0].item(),
                    "class_id": person_class_id,
                }
                all_detections_to_draw.append(person_detection)

                person_crop: np.ndarray = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                epi_results = epi_detector.detect(person_crop)

                epi_detections_on_person: List[DetectionDict] = []
                for epi_res in epi_results:
                    for epi_box in epi_res.boxes:
                        class_id: int = int(epi_box.cls)

                        if class_id == epi_model_person_class_id:
                            continue

                        ex1_rel, ey1_rel, ex2_rel, ey2_rel = [
                            int(i) for i in epi_box.xyxy[0]
                        ]

                        ex1_abs, ey1_abs, ex2_abs, ey2_abs = (
                            ex1_rel + x1,
                            ey1_rel + y1,
                            ex2_rel + x1,
                            ey2_rel + y1,
                        )

                        epi_detection: DetectionDict = {
                            "name": epi_class_names[class_id],
                            "box": [ex1_abs, ey1_abs, ex2_abs, ey2_abs],
                            "confidence": epi_box.conf[0].item(),
                            "class_id": class_id,
                        }
                        all_detections_to_draw.append(epi_detection)
                        epi_detections_on_person.append(epi_detection)

                alerts_for_person = evaluator.check_compliance(epi_detections_on_person)
                if alerts_for_person:
                    all_alerts.extend(alerts_for_person)

        if all_alerts and (time.time() - last_notification_time) > 15:
            full_alert_message = "🚨 ALERTA DE NÃO CONFORMIDADE 🚨\n" + "\n".join(
                all_alerts
            )
            print(full_alert_message)
            notifier.send_alert(full_alert_message)
            last_notification_time = time.time()

        frame = draw_detections(frame, all_detections_to_draw)
        frame = draw_compliance_status(frame, all_alerts)

        resized_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("EPI Monitor", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
