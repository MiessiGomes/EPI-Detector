import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, webhook_url: Optional[str]) -> None:
        self.webhook_url = webhook_url

    def send_alert(self, message: str, image_path: Optional[Path] = None) -> None:
        """
        Sends a notification message to the configured Discord webhook.
        If an image_path is provided, it sends the image along with the message.
        """
        if not self.webhook_url:
            logger.warning("Discord webhook URL not set. Skipping notification.")
            return

        payload = {"content": message}

        try:
            if image_path and image_path.is_file():
                with open(image_path, "rb") as f:
                    files = {"file": (image_path.name, f, "image/jpeg")}
                    response = requests.post(
                        self.webhook_url, data=payload, files=files
                    )
            else:
                response = requests.post(self.webhook_url, json=payload)

            response.raise_for_status()  # Raises an HTTPError for bad responses
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Discord alert: {e}")
