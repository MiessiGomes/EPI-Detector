import logging
from typing import Any, Dict, List

import cv2

from epi_monitor.config.logging_config import setup_logging
from epi_monitor.config.settings import SETTINGS
from epi_monitor.core.detector import Detector
from epi_monitor.core.evaluator import Evaluator
from epi_monitor.core.notifications import handle_notifications
from epi_monitor.core.notifier import Notifier
from epi_monitor.core.processing import process_person_track
from epi_monitor.utils.draw_utils import draw_detections, draw_final_results

DetectionDict = Dict[str, Any]


def main() -> None:
    # --- Setup Logging ---
    setup_logging()
    logger = logging.getLogger(__name__)

    # --- Initialization ---
    logger.info("Initializing application components...")
    person_detector = Detector(SETTINGS.MODEL_PERSON_PATH)
    person_class_id: int = next(
        cid
        for cid, name in person_detector.model.names.items()
        if name == SETTINGS.PERSON_CLASS_NAME
    )

    epi_detector = Detector(SETTINGS.EPI_DETECTOR_PATH)
    epi_class_names: Dict[int, str] = epi_detector.model.names
    epi_model_person_class_id: int = next(
        (
            cid
            for cid, name in epi_class_names.items()
            if name == SETTINGS.EPI_MODEL_IGNORE_CLASS_NAME
        ),
        None,
    )

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
        logger.critical(f"Error: Could not open video source: {SETTINGS.VIDEO_SOURCE}")
        return

    logger.info("Starting 2-Stage EPI Monitor with Re-ID...")
    notified_track_ids: Dict[int, float] = {}

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break

        person_results = person_detector.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            conf=SETTINGS.PERSON_CONFIDENCE_THRESHOLD,
        )

        all_detections: List[DetectionDict] = []
        non_compliance_by_track_id: Dict[int, List[str]] = {}

        if person_results[0].boxes.id is not None:
            for box in person_results[0].boxes:
                if int(box.cls) != person_class_id:
                    continue

                detections, alerts, track_id = process_person_track(
                    box,
                    frame,
                    epi_detector,
                    evaluator,
                    epi_class_names,
                    epi_model_person_class_id,
                )
                all_detections.extend(detections)
                if alerts:
                    non_compliance_by_track_id[track_id] = alerts

        # --- Draw detections first, so the logged image has them ---
        frame_with_detections = draw_detections(frame.copy(), all_detections)

        # --- Handle notifications and logging ---
        all_display_alerts: List[str] = []
        for track_id, alerts in non_compliance_by_track_id.items():
            display_alerts = handle_notifications(
                frame_with_detections,
                alerts,
                track_id,
                notified_track_ids,
                notifier,
            )
            all_display_alerts.extend(display_alerts)

        # --- Draw final results on the original frame for display ---
        final_frame = draw_final_results(frame, all_detections, all_display_alerts)
        cv2.imshow("EPI Monitor", final_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Quit key pressed. Exiting application.")
            break

    # --- Cleanup ---
    logger.info("Releasing resources.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
