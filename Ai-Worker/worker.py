import time
import threading
import cv2
import requests
import torch
from ultralytics import YOLO
import os

# --- Import your custom modules ---
from ingestion_funnel import start_funnel, frame_queues, CAMERAS
from plugins.tussle_slowfast import TusslePlugin
from plugins.fall_rule_based import RuleBasedFallPlugin
from video_recorder import ClipRecorder

# ==========================================
# 1. CONFIGURATION
# ==========================================
ENV = os.environ.get("ENV", "development").lower()
# Corrected port to 3000 to match your index.js
ALERT_CREATE_URL = os.environ.get("ALERT_CREATE_URL", "http://localhost:5000/api/alerts")
camera_ids = [cam["id"] for cam in CAMERAS]

# Alert Cooldowns (1 alert per event type per camera every 5 seconds)
alert_cooldowns = {cam_id: {"fall": 0, "tussle": 0} for cam_id in camera_ids}

# Fall persistence: (cam_id, track_id) → last_fall_time
fall_persist = {}

def display_loop(stop_event, display_frames, display_lock):
    """Dedicated display loop so rendering does not block AI processing."""
    while not stop_event.is_set():
        has_frame = False
        with display_lock:
            items = list(display_frames.items())

        for cam_id, frame in items:
            if frame is None:
                continue
            has_frame = True
            cv2.imshow(f"Tracked Output {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Display] Window closed by user.")
            stop_event.set()
            break

        if not has_frame:
            time.sleep(0.005)

    cv2.destroyAllWindows()

# ==========================================
# 2. THE SHARED BACKBONE (Geometry Extractor)
# ==========================================
class SharedBackbone:
    def __init__(self):
        print("🧠 Initializing Shared Backbone (YOLOv8-Pose)...")
        self.model = YOLO("yolov8n-pose.pt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"✅ Backbone active on device: {self.device.upper()}")

    def process_batch(self, current_batch):
        batch_frames = []
        batch_cam_ids = []

        for cam_id, frame in current_batch.items():
            batch_frames.append(frame)
            batch_cam_ids.append(cam_id)

        if not batch_frames:
            return {}

        results = self.model.track(
            batch_frames, 
            persist=True, 
            classes=[0], 
            verbose=False,
            tracker="botsort.yaml" 
        )

        scene_state = {}
        for i, result in enumerate(results):
            cam_id = batch_cam_ids[i]
            frame = batch_frames[i]
            
            scene_state[cam_id] = {
                "raw_frame": frame,
                "tracked_people": {}
            }

            if result.boxes is None or result.boxes.id is None:
                continue

            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
            if has_keypoints:
                keypoints = result.keypoints.xy.cpu().numpy()

            for idx, track_id in enumerate(track_ids):
                bbox = boxes[idx]
                person_data = {
                    "bbox": bbox,        
                    "confidence": float(confs[idx]),
                    "keypoints": keypoints[idx] if has_keypoints else None,
                    "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                }
                scene_state[cam_id]["tracked_people"][track_id] = person_data

        return scene_state

# ==========================================
# 3. EGRESS CONTROLLER (Webhooks & Saving)
# ==========================================
def send_webhook(alert_data, raw_frame):
    """
    CRITICAL FIX: Maps internal variables to exact Sequelize Model keys.
    Database expects: alertType, snapshotUrl, videoUrl
    """
    cam_id = alert_data["cameraId"]
    event_type = alert_data["eventType"]
    current_time = alert_data["timestamp"] 
    
    # Check Cooldown
    if current_time - alert_cooldowns[cam_id][event_type] < 5.0:
        return
    alert_cooldowns[cam_id][event_type] = current_time

    # Save Snapshot
    filename = f"{cam_id}_{event_type}_{current_time}.jpg"
    filepath = f"../web-dashboard/public/alerts/{filename}" 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, raw_frame)
    
    # Payload keys MUST match your Alert.js definition exactly
    payload = {
        "cameraId": str(cam_id), 
        "alertType": event_type,              # FIXED: Match alert.controller check
        "message": f"Security Alert: {event_type.upper()} detected on {cam_id}.",
        "snapshotUrl": f"/alerts/{filename}",  # FIXED: Match Sequelize Model
        "videoUrl": alert_data.get("videoPath"), # FIXED: Match Sequelize Model
        "severity": "High" if event_type in ["fall", "tussle"] else "Medium",
        "status": "Open"
    }
    
    try:
        response = requests.post(ALERT_CREATE_URL, json=payload)
        if response.ok:
            print(f"🚨 ALERT SAVED: {event_type.upper()} on {cam_id}")
        else:
            print(f"⚠️ Alert API responded with {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Failed to call alert create endpoint: {e}")

# ==========================================
# 4. MASTER EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    start_funnel()
    backbone = SharedBackbone()

    stop_event = threading.Event()
    display_lock = threading.Lock()
    display_frames = {cam_id: None for cam_id in camera_ids}
    display_thread = None
    
    if ENV == "development":
        display_thread = threading.Thread(
            target=display_loop,
            args=(stop_event, display_frames, display_lock),
            daemon=True,
        )
        display_thread.start()
    
    plugins = [
        TusslePlugin(model_path="./best_slowfast_fight_model (2).pth", camera_ids=camera_ids)
    ]

    recorder = ClipRecorder(before_frames=30, after_frames=30, fps=10)
    
    print("🚀 Master Loop Online. Analyzing streams...")
    
    try:
        while not stop_event.is_set():
            current_batch = {}
            for cam_id, q in frame_queues.items():
                if not q.empty():
                    current_batch[cam_id] = q.get()
            
            if not current_batch:
                time.sleep(0.01)
                continue
                
            scene_state = backbone.process_batch(current_batch)
            if not scene_state:
                continue

            for cam_id, data in scene_state.items():
                recorder.update_frame(cam_id, data["raw_frame"])

            for plugin in plugins:
                if isinstance(plugin, TusslePlugin):
                    tussle_input = {cid: s["raw_frame"] for cid, s in scene_state.items()}
                    alerts = plugin.process_batch(tussle_input)
                else:
                    alerts = plugin.process_batch(scene_state)

                for alert in alerts:
                    trigger_cam = alert["cameraId"]

                    if alert["eventType"] == "fall":
                        key = (trigger_cam, alert["trackId"])
                        fall_persist[key] = time.time()

                    trigger_frame = scene_state[trigger_cam]["raw_frame"]
                    timestamp = int(time.time())
                    
                    video_path = recorder.trigger(trigger_cam, alert["eventType"], timestamp)
                    
                    alert["timestamp"] = timestamp
                    alert["videoPath"] = video_path
                    
                    send_webhook(alert, trigger_frame)

            # Drawing Logic
            for cam_id, state in scene_state.items():
                frame = state["raw_frame"]
                for track_id, person in state["tracked_people"].items():
                    x1, y1, x2, y2 = map(int, person["bbox"])
                    key = (cam_id, track_id)
                    
                    is_fall = False
                    if key in fall_persist:
                        if time.time() - fall_persist[key] < 3.0:
                            is_fall = True
                        else:
                            del fall_persist[key]

                    color = (0, 0, 255) if is_fall else (0, 255, 0)
                    label = f"FALL ID {track_id}" if is_fall else f"ID {track_id}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if ENV == "development":
                with display_lock:
                    for cam_id, state in scene_state.items():
                        display_frames[cam_id] = state["raw_frame"].copy()
                    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
    finally:
        stop_event.set()
        if display_thread is not None and display_thread.is_alive():
            display_thread.join(timeout=1.0)
        cv2.destroyAllWindows()