import os
import threading
import time

import cv2

from ingestion_funnel import CAMERA_CONFIG, frame_queues, get_ingestion_service, start_funnel
from plugins.tussle_slowfast import TusslePlugin
from video_recorder import ClipRecorder
from worker_alerting import AlertDispatcher
from worker_backbone import SharedBackbone
from worker_display import display_loop


class WorkerRuntime:
    def __init__(self, env=None):
        self.env = (env or os.environ.get("ENV", "development")).lower()
        self.stop_event = threading.Event()
        self.display_lock = threading.Lock()
        self.display_frames = {}
        self.display_thread = None

        self.camera_ids = list(CAMERA_CONFIG.keys())
        self.backbone = None
        self.plugins = []
        self.recorder = None
        self.alert_dispatcher = AlertDispatcher()
        self.fall_persist = {}

    def setup(self):
        start_funnel()

        ingestion_service = get_ingestion_service()
        self.camera_ids = list(ingestion_service.camera_config.keys())

        self.backbone = SharedBackbone()

        if self.env == "development":
            self.display_frames = {cam_id: None for cam_id in self.camera_ids}
            self.display_thread = threading.Thread(
                target=display_loop,
                args=(self.stop_event, self.display_frames, self.display_lock),
                daemon=True,
            )
            self.display_thread.start()

        print("🔌 Loading AI Plugins...")
        self.plugins = [
            TusslePlugin(model_path="./best_slowfast_fight_model (2).pth", camera_ids=self.camera_ids)
        ]

        self.recorder = ClipRecorder(before_frames=200, after_frames=200, fps=10)
        print("🚀 Master Loop Online. Analyzing streams...")

    @staticmethod
    def _next_batch():
        return {cam_id: q.get() for cam_id, q in list(frame_queues.items()) if not q.empty()}

    def _run_plugins(self, scene_state):
        for plugin in self.plugins:
            if isinstance(plugin, TusslePlugin):
                tussle_input = {cam_id: state["raw_frame"] for cam_id, state in scene_state.items()}
                alerts = plugin.process_batch(tussle_input)
            else:
                alerts = plugin.process_batch(scene_state)

            for alert in alerts:
                trigger_cam = alert["cameraId"]

                if alert["eventType"] == "fall":
                    key = (trigger_cam, alert["trackId"])
                    self.fall_persist[key] = time.time()

                if trigger_cam not in scene_state:
                    continue

                trigger_frame = scene_state[trigger_cam]["raw_frame"]
                timestamp = int(time.time())
                video_path = self.recorder.trigger(trigger_cam, alert["eventType"], timestamp)

                alert["timestamp"] = timestamp
                alert["videoPath"] = video_path
                self.alert_dispatcher.send(alert, trigger_frame)

    def _annotate_frames(self, scene_state):
        now = time.time()
        for cam_id, state in scene_state.items():
            frame = state["raw_frame"]
            tracked = state["tracked_people"]

            for track_id, person in tracked.items():
                x1, y1, x2, y2 = map(int, person["bbox"])

                key = (cam_id, track_id)
                is_fall = False
                if key in self.fall_persist:
                    if now - self.fall_persist[key] < 3.0:
                        is_fall = True
                    else:
                        del self.fall_persist[key]

                if is_fall:
                    color = (0, 0, 255)
                    label = f"FALL ID {track_id}"
                else:
                    color = (0, 255, 0)
                    label = f"ID {track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

    def _publish_display_frames(self, scene_state):
        if self.env != "development":
            return

        with self.display_lock:
            for cam_id, state in scene_state.items():
                self.display_frames[cam_id] = state["raw_frame"].copy()

    def run(self):
        self.setup()

        try:
            while not self.stop_event.is_set():
                current_batch = self._next_batch()
                if not current_batch:
                    time.sleep(0.01)
                    continue

                scene_state = self.backbone.process_batch(current_batch)
                if not scene_state:
                    continue

                for cam_id, data in scene_state.items():
                    self.recorder.update_frame(cam_id, data["raw_frame"])

                self._run_plugins(scene_state)
                self._annotate_frames(scene_state)
                self._publish_display_frames(scene_state)

        except KeyboardInterrupt:
            print("\n🛑 Shutting down system...")
        finally:
            self.stop_event.set()
            if self.display_thread is not None and self.display_thread.is_alive():
                self.display_thread.join(timeout=1.0)
            cv2.destroyAllWindows()
