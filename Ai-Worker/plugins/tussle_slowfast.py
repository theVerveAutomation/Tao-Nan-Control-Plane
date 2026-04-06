import cv2
import torch
import numpy as np
import collections
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn

class TusslePlugin:
    def __init__(self, model_path, camera_ids):
        print("🔌 Initializing Tussle (SlowFast) Plugin...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Config from your Kaggle script ---
        self.CONFIDENCE_THRESHOLD = 0.55
        self.SUSTAINED_THRESHOLD = 3
        self.INFERENCE_INTERVAL = 15 # Run every 15 frames
        
        # --- Load Model ---
        self.model = slowfast_r50(pretrained=False)
        in_features = self.model.blocks[6].proj.in_features
        self.model.blocks[6].proj = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features, 2)
        )
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)
        self.model.eval()

        # --- Independent State Management per Camera ---
        self.camera_ids = camera_ids
        self.frame_buffers = {cam: collections.deque(maxlen=64) for cam in camera_ids}
        self.prediction_history = {cam: collections.deque(maxlen=5) for cam in camera_ids}
        self.suspicious_counts = {cam: 0 for cam in camera_ids}
        self.frame_counters = {cam: 0 for cam in camera_ids}

    def process_batch(self, batched_frames_dict):
        """
        Takes a dictionary of {cam_id: latest_rgb_frame} from the main loop.
        Returns a list of alerts if a tussle is detected.
        """
        alerts = []
        cams_ready_for_inference = []
        fast_tensors = []
        slow_tensors = []
        #print(f"[TusslePlugin][DEBUG] Received batch: {list(batched_frames_dict.keys())}")

        # 1. Update buffers for all cameras
        for cam_id, frame in batched_frames_dict.items():
            #print(f"[TusslePlugin][DEBUG] cam_id={cam_id} frame type={type(frame)} shape={getattr(frame, 'shape', None)}")
            if frame is None:
                print(f"[TusslePlugin][DEBUG] cam_id={cam_id} received None frame, skipping.")
                continue
            # Preprocess to 224x224 RGB as your script requires
            import numpy as np
            if frame is None or not isinstance(frame, np.ndarray):
                print(f"[TusslePlugin][DEBUG] cam_id={cam_id} invalid frame (not ndarray), skipping batch.")
                return []
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (224, 224))
            self.frame_buffers[cam_id].append(resized)
            #print(f"[TusslePlugin][DEBUG] cam_id={cam_id} buffer size={len(self.frame_buffers[cam_id])}")
            self.frame_counters[cam_id] += 1
            # Check if this camera is ready for inference
            if len(self.frame_buffers[cam_id]) == 64 and self.frame_counters[cam_id] % self.INFERENCE_INTERVAL == 0:
                print(f"[TusslePlugin][DEBUG] Inference triggered for cam_id={cam_id} at frame_counter={self.frame_counters[cam_id]}")
                cams_ready_for_inference.append(cam_id)
                # Extract Fast/Slow pathways
                clip = np.array(self.frame_buffers[cam_id])
                fast = clip[::2][:32]
                slow = fast[::4][:8]
                # Format for PyTorch (C, T, H, W)
                fast_t = torch.from_numpy(fast).permute(3, 0, 1, 2).float() / 255.0
                slow_t = torch.from_numpy(slow).permute(3, 0, 1, 2).float() / 255.0
                fast_tensors.append(fast_t)
                slow_tensors.append(slow_t)

        # 2. Run Batched Inference
        if len(cams_ready_for_inference) > 0:
            #print(f"[TusslePlugin][DEBUG] Running inference for cams: {cams_ready_for_inference}")
            # Stack into (Batch, C, T, H, W)
            batch_fast = torch.stack(fast_tensors).to(self.device)
            batch_slow = torch.stack(slow_tensors).to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    preds = self.model([batch_slow, batch_fast])
                    probabilities = torch.softmax(preds, dim=1)
            print(f"[TusslePlugin][DEBUG] Model probabilities: {probabilities.cpu().numpy()}")
            # 3. Apply your Grace Period Logic
            for idx, cam_id in enumerate(cams_ready_for_inference):
                fight_score = probabilities[idx][1].item()
                print(f"[TusslePlugin][DEBUG] cam_id={cam_id} fight_score={fight_score:.4f}")
                self.prediction_history[cam_id].append(fight_score)

                if len(self.prediction_history[cam_id]) == 5:
                    smoothed_score = sum(self.prediction_history[cam_id]) / 5.0
                    print(f"[TusslePlugin][DEBUG] cam_id={cam_id} smoothed_score={smoothed_score:.4f} suspicious_count={self.suspicious_counts[cam_id]}")
                    # --- YOUR EXACT GRACE PERIOD LOGIC ---
                    if smoothed_score >= self.CONFIDENCE_THRESHOLD:
                        #print(f"[TusslePlugin][DEBUG] cam_id={cam_id} smoothed_score above threshold, incrementing suspicious_count.")
                        self.suspicious_counts[cam_id] += 1
                    else:
                        if self.suspicious_counts[cam_id] > 0:
                            #print(f"[TusslePlugin][DEBUG] cam_id={cam_id} smoothed_score below threshold, decrementing suspicious_count.")
                            self.suspicious_counts[cam_id] -= 1 

                    # Trigger Alarm
                    if self.suspicious_counts[cam_id] >= self.SUSTAINED_THRESHOLD:
                        #print(f"⚠️  Tussle detected on {cam_id} with smoothed confidence {smoothed_score:.2f}")
                        alerts.append({
                            "cameraId": cam_id,
                            "eventType": "tussle",
                            "confidence": smoothed_score
                        })
                        # Debounce
                        self.suspicious_counts[cam_id] = 0
                        self.prediction_history[cam_id].clear()

        return alerts