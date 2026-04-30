import cv2
import torch
import collections
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn
import torchvision.transforms as T

class TusslePlugin:
    def __init__(self, model_path, camera_ids):
        print("🔌 Initializing Tussle (SlowFast) Plugin...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- SEVERITY MATRIX ---
        # Scaling both Confidence and Sustained counts together
        self.SEVERITY_THRESHOLDS = {
            "CRITICAL": {"conf": 0.95, "sustained": 1},
            "HIGH":     {"conf": 0.85, "sustained": 2},
            "MEDIUM":   {"conf": 0.75, "sustained": 2},
            "LOW":      {"conf": 0.65, "sustained": 4}
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

        # --- GPU-Accelerated Preprocessing Pipeline ---
        # 1. ToTensor() converts to [C, H, W] in the range [0.0, 1.0]
        # 2. Resize uses the GPU to downscale to 224x224
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Resize((224, 224), antialias=True)
        ])

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
                
            # Convert BGR to RGB 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 👇 GPU ACCELERATION: Convert to tensor and resize on GPU immediately
            # Shape is [C, H, W]
            tensor_frame = self.transform(rgb_frame).to(self.device)
            
            self.frame_buffers[cam_id].append(tensor_frame)
            self.frame_counters[cam_id] += 1
            
            if len(self.frame_buffers[cam_id]) == 64 and self.frame_counters[cam_id] % self.INFERENCE_INTERVAL == 0:
                cams_ready_for_inference.append(cam_id)
                
                # Stack the 64 tensors. Resulting shape: [T, C, H, W]
                clip = torch.stack(list(self.frame_buffers[cam_id]))
                
                # Slicing the tensors along the Temporal (T) dimension
                fast_t = clip[::2][:32]
                slow_t = fast_t[::4][:8]
                
                # SlowFast requires shape: [C, T, H, W]
                # Currently they are [T, C, H, W], so we permute dim 0 and 1
                fast_t = fast_t.permute(1, 0, 2, 3) 
                slow_t = slow_t.permute(1, 0, 2, 3)

                # Add to batch and cast to FP16 to match your autocast setting!
                fast_tensors.append(fast_t.half())
                slow_tensors.append(slow_t.half())

        # 2. Run Batched Inference
        if len(cams_ready_for_inference) > 0:
            # batch shape: [Batch_Size, C, T, H, W]
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