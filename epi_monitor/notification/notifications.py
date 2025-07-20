import logging
import time
from typing import Dict, List

import numpy as np

from epi_monitor.notification.notifier import Notifier
from epi_monitor.utils.event_logger import log_event

logger = logging.getLogger(__name__)
NOTIFICATION_COOLDOWN_S: int = 30


def handle_notifications(
    frame_with_detections: np.ndarray,
    alerts: List[str],
    track_id: int,
    notified_track_ids: Dict[int, float],
    notifier: Notifier,
) -> List[str]:
    """
    Manages notification cooldowns, logs events, and sends alerts with images.
    """
    if not alerts:
        return []

    current_time = time.time()
    last_notified_time = notified_track_ids.get(track_id, 0)

    if (current_time - last_notified_time) > NOTIFICATION_COOLDOWN_S:
        # Log the event and get the path to the saved image
        image_filepath = log_event(frame_with_detections, track_id)

        # Send notification with the image
        alert_message = f"ID {track_id}: {alerts[0]}"
        logger.info(f"Sending alert for track ID {track_id}: {alerts[0]}")
        notifier.send_alert(
            f"🚨 ALERTA DE NÃO CONFORMIDADE 🚨\n{alert_message}",
            image_path=image_filepath,
        )
        notified_track_ids[track_id] = current_time
        return [alert_message]

    return []
