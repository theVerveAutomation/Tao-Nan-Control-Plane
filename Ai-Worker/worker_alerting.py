import os
from collections import defaultdict

import cv2
import requests

from config import ALERT_COOLDOWN_SECONDS, ALERT_CREATE_URL


class AlertDispatcher:
    def __init__(self, alert_create_url=None):
        self.alert_create_url = alert_create_url or ALERT_CREATE_URL
        self.alert_cooldowns = defaultdict(lambda: {"fall": 0, "tussle": 0})

    def send(self, alert_data, raw_frame):
        cam_id = alert_data["cameraId"]
        event_type = alert_data["eventType"]
        current_time = alert_data["timestamp"]

        if current_time - self.alert_cooldowns[cam_id][event_type] < ALERT_COOLDOWN_SECONDS:
            return
        self.alert_cooldowns[cam_id][event_type] = current_time

        filename = f"{cam_id}_{event_type}_{current_time}.jpg"
        filepath = f"../web-dashboard/public/alerts/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, raw_frame)

        payload = {
            "cameraId": cam_id,
            "eventType": event_type,
            "confidence": alert_data["confidence"],
            "imagePath": f"/alerts/{filename}",
            "videoPath": alert_data.get("videoPath"),
        }

        try:
            response = requests.post(self.alert_create_url, json=payload)
            if response.ok:
                print(f"🚨 ALERT SAVED: {event_type.upper()} on {cam_id} -> {self.alert_create_url}")
            else:
                print(f"⚠️ Alert API responded with {response.status_code}: {response.text[:200]}")
        except Exception as exc:
            print(f"⚠️ Failed to call alert create endpoint ({self.alert_create_url}): {exc}")
