from typing import Optional

import requests


class Notifier:
    def __init__(self, webhook_url: Optional[str]) -> None:
        self.webhook_url = webhook_url

    def send_alert(self, message: str) -> None:
        """
        Sends a notification message to the configured Discord webhook.
        """
        if not self.webhook_url:
            print("Discord webhook URL not set. Skipping notification.")
            return

        payload = {"content": message}
        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error sending Discord alert: {e}")
