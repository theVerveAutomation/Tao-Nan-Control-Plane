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
        
        # --- SEVERITY MATRIX ---
        # Scaling both Confidence and Sustained counts together
        self.SEVERITY_THRESHOLDS = {
            "CRITICAL": {"conf": 0.85, "sustained": 4},
            "HIGH":     {"conf": 0.75, "sustained": 3},
            "MEDIUM":   {"conf": 0.65, "sustained": 2},
            "LOW":      {"conf": 0.55, "sustained": 1}
        }
        
        # The lowest baseline needed to even start the suspicious counter
        self.BASE_CONF_THRESHOLD = self.SEVERITY_THRESHOLDS["LOW"]["conf"]
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

    def _determine_severity(self, smoothed_score, count):
        """
        Checks the ladder from top to bottom. Returns the highest valid severity.
        """
        if smoothed_score >= self.SEVERITY_THRESHOLDS["CRITICAL"]["conf"] and count >= self.SEVERITY_THRESHOLDS["CRITICAL"]["sustained"]:
            return "CRITICAL"
        elif smoothed_score >= self.SEVERITY_THRESHOLDS["HIGH"]["conf"] and count >= self.SEVERITY_THRESHOLDS["HIGH"]["sustained"]:
            return "HIGH"
        elif smoothed_score >= self.SEVERITY_THRESHOLDS["MEDIUM"]["conf"] and count >= self.SEVERITY_THRESHOLDS["MEDIUM"]["sustained"]:
            return "MEDIUM"
        elif smoothed_score >= self.SEVERITY_THRESHOLDS["LOW"]["conf"] and count >= self.SEVERITY_THRESHOLDS["LOW"]["sustained"]:
            return "LOW"
            
        return None

    def process_batch(self, batched_frames_dict):
        alerts = []
        cams_ready_for_inference = []
        fast_tensors = []
        slow_tensors = []

        # 1. Update buffers for all cameras
        for cam_id, frame in batched_frames_dict.items():
            #print(f"[TusslePlugin][DEBUG] cam_id={cam_id} frame type={type(frame)} shape={getattr(frame, 'shape', None)}")
            if frame is None:
                continue
            
            import numpy as np
            if not isinstance(frame, np.ndarray):
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (224, 224))
            self.frame_buffers[cam_id].append(resized)
            self.frame_counters[cam_id] += 1
            
            if len(self.frame_buffers[cam_id]) == 64 and self.frame_counters[cam_id] % self.INFERENCE_INTERVAL == 0:
                cams_ready_for_inference.append(cam_id)
                clip = np.array(self.frame_buffers[cam_id])
                fast = clip[::2][:32]
                slow = fast[::4][:8]
                fast_t = torch.from_numpy(fast).permute(3, 0, 1, 2).float() / 255.0
                slow_t = torch.from_numpy(slow).permute(3, 0, 1, 2).float() / 255.0
                fast_tensors.append(fast_t)
                slow_tensors.append(slow_t)

        # 2. Run Batched Inference
        if len(cams_ready_for_inference) > 0:
            batch_fast = torch.stack(fast_tensors).to(self.device)
            batch_slow = torch.stack(slow_tensors).to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    preds = self.model([batch_slow, batch_fast])
                    probabilities = torch.softmax(preds, dim=1)
                    
            # 3. Apply Dynamic Severity Ladder
            for idx, cam_id in enumerate(cams_ready_for_inference):
                fight_score = probabilities[idx][1].item()
                self.prediction_history[cam_id].append(fight_score)

                if len(self.prediction_history[cam_id]) == 5:
                    smoothed_score = sum(self.prediction_history[cam_id]) / 5.0
                    
                    # Increment counter if we meet the absolute minimum baseline (LOW threshold)
                    if smoothed_score >= self.BASE_CONF_THRESHOLD:
                        self.suspicious_counts[cam_id] += 1
                    else:
                        if self.suspicious_counts[cam_id] > 0:
                            self.suspicious_counts[cam_id] -= 1 

                    # Determine severity using the matrix
                    severity = self._determine_severity(smoothed_score, self.suspicious_counts[cam_id])

                    if severity:
                        print(f"⚠️ [{cam_id}] TUSSLE DETECTED! Severity: {severity} | Conf: {smoothed_score:.2f} | Count: {self.suspicious_counts[cam_id]}")
                        alerts.append({
                            "cameraId": cam_id,
                            "eventType": "tussle",
                            "confidence": smoothed_score,
                            "severity": severity
                        })
                        
                        # Once triggered, reset the state so we don't spam the database
                        self.suspicious_counts[cam_id] = 0
                        self.prediction_history[cam_id].clear()

        return alerts