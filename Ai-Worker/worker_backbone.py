import torch
from ultralytics import YOLO

from config import YOLO_MODEL_PATH


class SharedBackbone:
    def __init__(self):
        print("🧠 Initializing Shared Backbone (YOLOv8-Pose)...")
        self.model = YOLO(YOLO_MODEL_PATH)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            tracker="botsort.yaml",
        )

        scene_state = {}

        for i, result in enumerate(results):
            cam_id = batch_cam_ids[i]
            frame = batch_frames[i]

            scene_state[cam_id] = {"raw_frame": frame, "tracked_people": {}}

            if result.boxes is None or result.boxes.id is None:
                continue

            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            has_keypoints = hasattr(result, "keypoints") and result.keypoints is not None
            if has_keypoints:
                keypoints = result.keypoints.xy.cpu().numpy()

            for idx, track_id in enumerate(track_ids):
                bbox = boxes[idx]
                scene_state[cam_id]["tracked_people"][track_id] = {
                    "bbox": bbox,
                    "confidence": float(confs[idx]),
                    "keypoints": keypoints[idx] if has_keypoints else None,
                    "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                }

        return scene_state
