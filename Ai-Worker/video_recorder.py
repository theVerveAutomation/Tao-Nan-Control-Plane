import cv2
import time
import threading
import os
from collections import deque

class ClipRecorder:
    def __init__(self, before_frames=30, after_frames=30, fps=10):
        print("🎥 Initializing Clip Recorder (Rolling Dashcam Buffer)...")
        self.before_frames = before_frames
        self.after_frames = after_frames
        self.fps = fps # Should roughly match your AI processing speed
        
        # State Management
        self.buffers = {}       # Holds the continuous 'Before' frames: { cam_id: deque }
        self.recordings = {}    # Holds the active recordings: { cam_id: dict }
        
        # Ensure the output directory exists
        self.output_dir = "../web-dashboard/public/alerts/clips"
        os.makedirs(self.output_dir, exist_ok=True)

    def update_frame(self, cam_id, frame):
        """Called every cycle to keep the rolling buffer fresh."""
        if frame is None: return

        # 1. Initialize the rolling buffer for new cameras
        if cam_id not in self.buffers:
            self.buffers[cam_id] = deque(maxlen=self.before_frames)

        # 2. Check if we are currently recording the 'After' segment for an alert
        if cam_id in self.recordings:
            self.recordings[cam_id]['frames'].append(frame)
            self.recordings[cam_id]['remaining'] -= 1

            # If the 'After' time is finished, save the video!
            if self.recordings[cam_id]['remaining'] <= 0:
                self._save_clip(cam_id)
        else:
            # 3. If no alert is happening, just keep rolling the 'Before' buffer
            self.buffers[cam_id].append(frame)

    def trigger(self, cam_id, event_type, timestamp):
        """Called by your plugins when a fall or tussle is detected."""
        # If we are already recording an event for this camera, don't interrupt it
        if cam_id in self.recordings:
            return None

        filename = f"{cam_id}_{event_type}_{timestamp}.mp4"
        
        # Grab the entire history of the 'Before' buffer
        current_history = list(self.buffers[cam_id])
        
        # Move it into the active recordings dictionary
        self.recordings[cam_id] = {
            'frames': current_history,
            'remaining': self.after_frames,
            'filename': filename
        }
        
        return f"/alerts/clips/{filename}"

    def _save_clip(self, cam_id):
        """Prepares the data and hands it to a background thread."""
        rec_data = self.recordings.pop(cam_id)
        frames = rec_data['frames']
        filepath = os.path.join(self.output_dir, rec_data['filename'])

        # START A BACKGROUND THREAD!
        # This prevents the AI loop from freezing while OpenCV compiles the .mp4
        threading.Thread(target=self._write_video, args=(frames, filepath), daemon=True).start()

    def _write_video(self, frames, filepath):
        """The actual CPU-heavy compilation (runs invisibly in the background)."""
        if not frames: return
        
        # Read the dimensions of the first frame
        h, w, _ = frames[0].shape
        
        # Use mp4v codec for standard .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
        
        for f in frames:
            out.write(f)
            
        out.release()
        print(f"🎬 VIDEO SAVED: {filepath}")