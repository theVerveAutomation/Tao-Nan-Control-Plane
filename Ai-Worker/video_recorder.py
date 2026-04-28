import cv2
import time
import threading
import os
import numpy as np
from collections import deque


class ClipRecorder:
    def __init__(self, before_frames=30, after_frames=30, fps=10):
        print("🎥 Initializing Clip Recorder (Rolling Dashcam Buffer)...")
        self.before_frames = before_frames
        self.after_frames = after_frames
        self.fps = fps

        # State Management
        self.buffers = {}     # Rolling before-buffers: { cam_id: deque }
        self.recordings = {}  # Active recordings: { recording_key: dict }

        # --- SHARED STORAGE PATH ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.shared_base = os.path.abspath(
            os.path.join(current_dir, "..", "..", "shared_storage", "alerts")
        )
        os.makedirs(self.shared_base, exist_ok=True)

    def update_frame(self, cam_id, frame):
        """Called every cycle to keep the rolling buffer fresh."""
        if frame is None:
            return

        # Initialize the rolling buffer for new cameras
        if cam_id not in self.buffers:
            self.buffers[cam_id] = deque(maxlen=self.before_frames)

        # Always keep rolling buffer updated even during recording
        # (fixes back-to-back alerts having empty before-buffers)
        self.buffers[cam_id].append(frame)

        # Feed frame to ALL active recordings for this camera
        keys_to_save = []
        for key, rec in self.recordings.items():
            if rec['cam_id'] == cam_id:
                rec['frames'].append(frame)
                rec['remaining'] -= 1
                if rec['remaining'] <= 0:
                    keys_to_save.append(key)

        for key in keys_to_save:
            self._save_clip(key)

    def trigger(self, cam_id, event_type, timestamp):
        """Called by plugins when a fall or tussle is detected."""
        # Unique key per alert — concurrent alerts on same camera never block each other
        recording_key = f"{cam_id}_{event_type}_{timestamp}"

        # Guard against exact duplicate keys (extremely unlikely)
        if recording_key in self.recordings:
            return None

        current_history = list(self.buffers.get(cam_id, []))
        print(f"🔔 Trigger fired for {cam_id} | type={event_type} | buffer={len(current_history)} before-frames")

        if len(current_history) < 5:
            print(f"⚠️  Buffer thin ({len(current_history)} frames) for {cam_id} — using whatever is available")

        filename = f"{cam_id}_{event_type}_{timestamp}.webm"

        self.recordings[recording_key] = {
            'cam_id': cam_id,
            'frames': current_history,
            'remaining': self.after_frames,
            'filename': filename,
        }

        return f"/alerts/{cam_id}/clips/{filename}"

    def _save_clip(self, recording_key):
        """Prepares the data and hands it to a background thread."""
        rec_data = self.recordings.pop(recording_key)
        frames = rec_data['frames']
        cam_id = rec_data['cam_id']

        cam_dir = os.path.join(self.shared_base, cam_id, "clips")
        os.makedirs(cam_dir, exist_ok=True)

        filepath = os.path.join(cam_dir, rec_data['filename'])

        threading.Thread(
            target=self._write_video,
            args=(frames, filepath),
            daemon=True
        ).start()

    def _write_video(self, frames, filepath):
        """Runs in background — compiles frames into a video file."""
        try:
            # Filter out None frames
            frames = [f for f in frames if f is not None]

            # Fallback to blank frame so a file is ALWAYS written
            if not frames:
                print(f"⚠️  No valid frames for {filepath} — writing blank placeholder")
                frames = [self._blank_frame()]

            # Normalize all frames to same resolution as first frame
            # (prevents crashes if camera reconnected mid-buffer at different resolution)
            h, w, _ = frames[0].shape
            frames = [cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f for f in frames]

            fourcc = cv2.VideoWriter_fourcc(*'vp80')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))

            if not out.isOpened():
                print(f"❌ VideoWriter failed to open: {filepath}")
                print("   → Check disk space, folder permissions, and vp80 codec availability.")
                return

            for f in frames:
                out.write(f)

            out.release()
            print(f"🎬 VIDEO SAVED: {len(frames)} frames → {filepath}")

        except Exception as e:
            print(f"❌ _write_video crashed for {filepath}: {e}")

    def _blank_frame(self, width=640, height=480):
        """Returns a black frame as last-resort fallback."""
        return np.zeros((height, width, 3), dtype=np.uint8)