import logging
import time
from typing import Dict, List

import numpy as np

from epi_monitor.config.settings import SETTINGS
from epi_monitor.db.database import insert_event
from epi_monitor.notification.notifier import Notifier
from epi_monitor.utils.event_logger import log_event

logger = logging.getLogger(__name__)
NOTIFICATION_COOLDOWN_S: int = 30


def handle_notifications(
    frame_with_detections: np.ndarray,
    missing_epis: List[str],
    present_epis: List[str],
    track_id: int,
    notified_track_ids: Dict[int, float],
    notifier: Notifier,
) -> List[str]:
    """
    Manages notifications, logs events, and saves them to the database,
    respecting a cooldown period for each tracked person.
    """
    if not missing_epis:
        return []

    current_time = time.time()
    last_notified_time = notified_track_ids.get(track_id, 0)

    if (current_time - last_notified_time) > NOTIFICATION_COOLDOWN_S:
        # 1. Log the event by saving an image
        image_filepath = log_event(frame_with_detections, track_id)
        notification_status = "SENT"

        # 2. Insert the event into the database
        insert_event(
            track_id=track_id,
            present_epis=present_epis,
            missing_epis=missing_epis,
            image_path=str(image_filepath) if image_filepath else None,
            notification_status=notification_status,
            camera_id=SETTINGS.CAMERA_ID,
        )

        # 3. Send the notification
        alert_message = f"ID {track_id}: Person is missing: {', '.join(missing_epis)}"
        logger.info(f"Sending alert for track ID {track_id}")
        notifier.send_alert(
            f"🚨 ALERTA DE NÃO CONFORMIDADE 🚨\n{alert_message}",
            image_path=image_filepath,
        )

        # 4. Update the timestamp to enforce cooldown
        notified_track_ids[track_id] = current_time

        return [alert_message]

    # If in cooldown, do nothing.
    return []
