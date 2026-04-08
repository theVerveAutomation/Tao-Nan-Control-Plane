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
NEXTJS_INGRESS_URL = "http://localhost:3000/api/ingress"
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
        self.model = YOLO("yolov8s-pose.pt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"✅ Backbone active on device: {self.device.upper()}")

    def process_batch(self, current_batch):
        """
        Takes the dict of raw frames, runs batched YOLO inference, 
        and parses the results into a clean 'Scene State' dictionary.
        """
        batch_frames = []
        batch_cam_ids = []

        for cam_id, frame in current_batch.items():
            batch_frames.append(frame)
            batch_cam_ids.append(cam_id)

        if not batch_frames:
            return {}

        # Batched Inference with BotSORT Tracking
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

            # If no people detected, skip parsing
            if result.boxes is None or result.boxes.id is None:
                continue

            # Move tensors to CPU memory
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
            if has_keypoints:
                keypoints = result.keypoints.xy.cpu().numpy()

            for idx, track_id in enumerate(track_ids):
                bbox = boxes[idx]

                # ✅ CHANGE 2: Removed green drawing from backbone entirely

                person_data = {
                    "bbox": bbox,        
                    "confidence": float(confs[idx]),
                    "keypoints": keypoints[idx] if has_keypoints else None,
                    "center": (
                        (bbox[0] + bbox[2]) / 2, 
                        (bbox[1] + bbox[3]) / 2  
                    )
                }
                scene_state[cam_id]["tracked_people"][track_id] = person_data

        return scene_state

# ==========================================
# 3. EGRESS CONTROLLER (Webhooks & Saving)
# ==========================================
def send_webhook(alert_data, raw_frame):
    """Saves the alert snapshot and posts the JSON payload to Next.js."""
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
    
    cv2.imwrite(filepath, raw_frame)
    
    payload = {
        "cameraId": cam_id,
        "eventType": event_type,
        "confidence": alert_data["confidence"],
        "imagePath": f"/alerts/{filename}",
        "videoPath": alert_data.get("videoPath") 
    }
    
    try:
        # requests.post(NEXTJS_INGRESS_URL, json=payload, timeout=2)
        print(f"🚨 WEBHOOK SENT: {event_type.upper()} on {cam_id} (Video compiling in background...)")
    except Exception as e:
        print(f"⚠️ Webhook failed to send: {e}")

# ==========================================
# 4. MASTER EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    
    # 1. Start the Ingestion Threads
    start_funnel()
    
    # 2. Initialize the Backbone
    backbone = SharedBackbone()

    # 2.1 Start display thread (development only)
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
    
    # 3. Initialize the Split Brain Plugins
    print("🔌 Loading AI Plugins...")
    plugins = [
        RuleBasedFallPlugin(confidence_threshold=0.75, aspect_ratio_threshold=1.2),
        #TusslePlugin(model_path="./best_slowfast_fight_model (2).pth", camera_ids=camera_ids)
    ]

    # Initialize the Video Recorder
    recorder = ClipRecorder(before_frames=30, after_frames=30, fps=10)
    
    print("🚀 Master Loop Online. Analyzing streams...")
    
    try:
        while not stop_event.is_set():
            # A. Gather the freshest frames
            current_batch = {}
            for cam_id, q in frame_queues.items():
                if not q.empty():
                    current_batch[cam_id] = q.get()
            
            if not current_batch:
                time.sleep(0.01)
                continue
                
            # B. Extract Geometry (Step 1: YOLO → scene_state)
            scene_state = backbone.process_batch(current_batch)
            if not scene_state:
                continue

            # Feed the raw frames into the rolling dashcam buffer
            for cam_id, data in scene_state.items():
                recorder.update_frame(cam_id, data["raw_frame"])

            # C. Pass state to all Plugins (Step 2: Plugin → alerts)
            for plugin in plugins:
                if isinstance(plugin, TusslePlugin):
                    tussle_input = {cam_id: state["raw_frame"] for cam_id, state in scene_state.items()}
                    alerts = plugin.process_batch(tussle_input)
                else:
                    alerts = plugin.process_batch(scene_state)

                # Process any triggered alerts
                for alert in alerts:
                    trigger_cam = alert["cameraId"]

                    # ✅ Update fall_persist with timestamp for this track
                    if alert["eventType"] == "fall":
                        key = (trigger_cam, alert["trackId"])
                        fall_persist[key] = time.time()

                    trigger_frame = scene_state[trigger_cam]["raw_frame"]
                    
                    timestamp = int(time.time())
                    video_path = recorder.trigger(trigger_cam, alert["eventType"], timestamp)
                    
                    alert["timestamp"] = timestamp
                    alert["videoPath"] = video_path
                    
                    send_webhook(alert, trigger_frame)

            # ✅ CHANGE 5: Draw RED/GREEN boxes after plugin loop
            for cam_id, state in scene_state.items():
                frame = state["raw_frame"]
                tracked = state["tracked_people"]

                for track_id, person in tracked.items():
                    x1, y1, x2, y2 = map(int, person["bbox"])

                    key = (cam_id, track_id)
                    is_fall = False
                    if key in fall_persist:
                        if time.time() - fall_persist[key] < 3.0:  # 🔥 show for 3 sec
                            is_fall = True
                        else:
                            del fall_persist[key]

                    if is_fall:
                        color = (0, 0, 255)   # 🔴 RED  — fall detected
                        label = f"FALL ID {track_id}"
                    else:
                        color = (0, 255, 0)   # 🟢 GREEN — normal
                        label = f"ID {track_id}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ✅ CHANGE 6: Copy to display_frames AFTER drawing (correct order)
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