import os
import time
import threading
from collections import defaultdict

import cv2
import requests

from config import ALERT_COOLDOWN_SECONDS, ALERT_CREATE_URL


class AlertDispatcher:
    def __init__(self, alert_create_url=None):
        self.alert_create_url = alert_create_url or ALERT_CREATE_URL
        self.alert_cooldowns = defaultdict(lambda: {"fall": 0, "tussle": 0})

        # --- SHARED STORAGE PATH (per-camera) ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.alerts_base = os.path.abspath(
            os.path.join(current_dir, "..", "..", "shared_storage", "alerts")
        )
        os.makedirs(self.alerts_base, exist_ok=True)

    def send(self, alert_data, raw_frame):
        cam_id = alert_data["cameraId"]
        event_type = alert_data["eventType"]
        current_time = alert_data["timestamp"]

        if current_time - self.alert_cooldowns[cam_id][event_type] < ALERT_COOLDOWN_SECONDS:
            return
        self.alert_cooldowns[cam_id][event_type] = current_time

        filename = f"{cam_id}_{event_type}_{current_time}.jpg"

        # Use per-camera snapshots directory
        snaps_dir = os.path.join(self.alerts_base, cam_id, "snapshots")
        os.makedirs(snaps_dir, exist_ok=True)
        filepath = os.path.join(snaps_dir, filename)
        cv2.imwrite(filepath, raw_frame)

        payload = {
            "cameraId": cam_id,
            "eventType": event_type,
            "confidence": alert_data["confidence"],
            "imagePath": f"/alerts/{cam_id}/snapshots/{filename}",
            "videoPath": alert_data.get("videoPath"),
        }

        # Fire the HTTP request in a background thread so the AI loop never stutters
        threading.Thread(target=self._post_alert, args=(payload,), daemon=True).start()

    def _post_alert(self, payload):
        """Background worker to handle network requests securely."""
        try:
            # Wait for the video writer background thread to finish writing the clip
            # before notifying the backend — prevents null video on fast detections
            time.sleep(22)

            response = requests.post(self.alert_create_url, json=payload, timeout=5)
            if response.ok:
                print(f"🚨 ALERT SAVED: {payload['eventType'].upper()} on {payload['cameraId']} -> {self.alert_create_url}")
            else:
                print(f"⚠️ Alert API responded with {response.status_code}: {response.text[:200]}")
        except Exception as exc:
            print(f"⚠️ Failed to call alert create endpoint ({self.alert_create_url}): {exc}")