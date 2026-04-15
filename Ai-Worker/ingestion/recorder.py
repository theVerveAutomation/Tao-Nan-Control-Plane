from collections import deque
from datetime import datetime
import os

import cv2


class EventRecorder:
    def __init__(self, camera_id, fps=30, pre_seconds=3, post_seconds=5, resolution=(640, 640)):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution

        self.pre_roll_frames = fps * pre_seconds
        self.buffer = deque(maxlen=self.pre_roll_frames)
        self.post_roll_frames = fps * post_seconds

        self.is_recording = False
        self.frames_left_to_record = 0

        # H.264 codec for web browser compatibility
        self.fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.writer = None

        # Ensure the output directory exists
        os.makedirs("./public/alerts", exist_ok=True)

    def update_buffer(self, frame):
        # Always maintain the short-term memory
        self.buffer.append(frame)

        # If an event triggered, write to disk
        if self.is_recording and self.writer is not None:
            # Resize frame to match writer resolution before writing
            resized_frame = cv2.resize(frame, self.resolution)
            self.writer.write(resized_frame)
            self.frames_left_to_record -= 1

            if self.frames_left_to_record <= 0:
                self.stop_recording()

    def trigger_event(self):
        if self.is_recording:
            self.frames_left_to_record = self.post_roll_frames
            return

        print(f"🚨 [{self.camera_id}] EVENT DETECTED! Saving pre-roll & recording...")
        self.is_recording = True
        self.frames_left_to_record = self.post_roll_frames

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"./public/alerts/{self.camera_id}_event_{timestamp}.mp4"

        self.writer = cv2.VideoWriter(self.filename, self.fourcc, self.fps, self.resolution)

        # Dump the pre-roll memory to the file immediately
        for buffered_frame in self.buffer:
            resized_frame = cv2.resize(buffered_frame, self.resolution)
            self.writer.write(resized_frame)

    def stop_recording(self):
        self.is_recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        print(f"💾 [{self.camera_id}] Recording saved: {self.filename}")
