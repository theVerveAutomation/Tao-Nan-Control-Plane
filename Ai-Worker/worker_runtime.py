import threading
import time

import cv2

from config import (
    CLIP_RECORDER_AFTER_FRAMES,
    CLIP_RECORDER_BEFORE_FRAMES,
    CLIP_RECORDER_FPS,
    ENABLE_RTMP_PUBLISH,
    ENV,
    RTMP_BASE_URL,
    RTMP_FFMPEG_BIN,
    RTMP_FFMPEG_PRESET,
    RTMP_FRAME_HEIGHT,
    RTMP_FRAME_WIDTH,
    RTMP_PUBLISH_FPS,
    RTMP_STREAM_PREFIX,
    RTMP_VIDEO_CODEC,
    TUSSLE_MODEL_PATH,
)
from ingestion_funnel import CAMERA_CONFIG, frame_queues, get_ingestion_service, start_funnel
from plugins.tussle_slowfast import TusslePlugin
from plugins.fall_rule_based import RuleBasedFallPlugin
from rtmp_publisher import RtmpPublisherHub, ffmpeg_exists
from video_recorder import ClipRecorder
from worker_alerting import AlertDispatcher
from worker_backbone import SharedBackbone


class WorkerRuntime:
    def __init__(self, env=None):
        self.env = (env or ENV).lower()
        self.stop_event = threading.Event()
        self.display_lock = threading.Lock()
        self.display_frames = {}
        self.display_thread = None
        self.opened_windows = set()

        self.camera_ids = list(CAMERA_CONFIG.keys())
        self.backbone = None
        self.plugins = []
        self.recorder = None
        self.alert_dispatcher = AlertDispatcher()
        self.fall_persist = {}
        self.rtmp_hub = RtmpPublisherHub(
            enabled=ENABLE_RTMP_PUBLISH,
            base_url=RTMP_BASE_URL,
            stream_prefix=RTMP_STREAM_PREFIX,
            fps=RTMP_PUBLISH_FPS,
            width=RTMP_FRAME_WIDTH,
            height=RTMP_FRAME_HEIGHT,
            ffmpeg_bin=RTMP_FFMPEG_BIN,
            video_codec=RTMP_VIDEO_CODEC,
            preset=RTMP_FFMPEG_PRESET,
        )

    @staticmethod
    def _next_batch():
        return {cam_id: q.get() for cam_id, q in list(frame_queues.items()) if not q.empty()}

    @staticmethod
    def _is_detection_enabled(cam_id):
        cam_cfg = CAMERA_CONFIG.get(cam_id, {})
        # Prefer the DB-driven `detections` flag (newer name). Fall back to
        # `detection` for backward compatibility, then `is_active`.
        return bool(
            cam_cfg.get("detections", cam_cfg.get("detection", cam_cfg.get("is_active", True)))
        )

    def _run_plugins(self, scene_state):
        enabled_scene_state = {
            cam_id: state for cam_id, state in scene_state.items() if self._is_detection_enabled(cam_id)
        }
        if not enabled_scene_state:
            return

        for plugin in self.plugins:
            if isinstance(plugin, TusslePlugin):
                # Tussle plugin keeps per-camera state; guard against unknown cameras.
                tussle_input = {
                    cam_id: state["raw_frame"]
                    for cam_id, state in enabled_scene_state.items()
                    if cam_id in getattr(plugin, "frame_buffers", {})
                }
                if not tussle_input:
                    continue
                alerts = plugin.process_batch(tussle_input)
            else:
                alerts = plugin.process_batch(enabled_scene_state)

            for alert in alerts:
                trigger_cam = alert["cameraId"]

                if alert["eventType"] == "fall":
                    key = (trigger_cam, alert["trackId"])
                    self.fall_persist[key] = time.time()

                if trigger_cam not in enabled_scene_state:
                    continue

                trigger_frame = enabled_scene_state[trigger_cam]["raw_frame"]
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
        # Skip local display when not in development or when RTMP publishing is active
        if self.env != "development" or ENABLE_RTMP_PUBLISH:
            return

        with self.display_lock:
            for cam_id, state in scene_state.items():
                # ensure display_frames contains the camera key (handles runtime-added cameras)
                if cam_id not in self.display_frames:
                    self.display_frames[cam_id] = None
                self.display_frames[cam_id] = state["raw_frame"].copy()

                # Auto-add camera id to runtime list if introduced at runtime
                if cam_id not in self.camera_ids:
                    print(f"🔔 [Runtime] New camera detected at runtime: {cam_id}. Adding to camera_ids.")
                    self.camera_ids.append(cam_id)

                    updated_plugins = []
                    # Update plugins that track camera_ids (best-effort) and try reinitialization
                    for plugin in self.plugins:
                        # Only add runtime-introduced cameras to plugins if
                        # detections are enabled for that camera.
                        if not self._is_detection_enabled(cam_id):
                            continue

                        if hasattr(plugin, "camera_ids") and isinstance(plugin.camera_ids, list):
                            if cam_id not in plugin.camera_ids:
                                plugin.camera_ids.append(cam_id)
                                updated_plugins.append(plugin.__class__.__name__)

                                # Call plugin hook if available to allow reinitialization for the new camera
                                try:
                                    if hasattr(plugin, "on_camera_added"):
                                        plugin.on_camera_added(cam_id)
                                    elif hasattr(plugin, "reinitialize"):
                                        plugin.reinitialize(cam_id)
                                except Exception:
                                    # non-fatal; proceed to next plugin
                                    pass

                    if updated_plugins:
                        print(f"🔧 [Runtime] Updated plugins for new camera {cam_id}: {', '.join(updated_plugins)}")

    def _publish_rtmp_streams(self, scene_state):
        if not ENABLE_RTMP_PUBLISH:
            return

        for cam_id, state in scene_state.items():
            frame = state.get("raw_frame")
            if frame is None:
                continue
            self.rtmp_hub.publish_frame(cam_id, frame)

        self.rtmp_hub.prune_removed_cameras(CAMERA_CONFIG.keys())

    def _render_display_windows(self):
        # Skip local window rendering when not in development or when RTMP publishing is active
        if self.env != "development" or ENABLE_RTMP_PUBLISH:
            return

        with self.display_lock:
            items = list(self.display_frames.items())

        for cam_id, frame in items:
            if frame is None:
                continue

            window_name = f"Tracked Output {cam_id}"
            if cam_id not in self.opened_windows:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    self.opened_windows.add(cam_id)
                    print(f"[Display] Opening window for camera {cam_id}")
                except Exception as exc:
                    print(f"[Display] Failed to create window for camera {cam_id}: {exc}")
                    continue

            try:
                cv2.imshow(window_name, frame)
            except Exception as exc:
                print(f"[Display] Failed to render frame for camera {cam_id}: {exc}")

        current_ids = set(CAMERA_CONFIG.keys())
        to_close = set(self.opened_windows) - current_ids
        if to_close:
            with self.display_lock:
                for cid in to_close:
                    try:
                        cv2.destroyWindow(f"Tracked Output {cid}")
                    except Exception:
                        pass
                    self.opened_windows.discard(cid)
                    self.display_frames.pop(cid, None)
                    print(f"[Display] Closed window for removed camera {cid}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Display] Window closed by user.")
            self.stop_event.set()

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
                self._publish_rtmp_streams(scene_state)
                self._publish_display_frames(scene_state)
                self._render_display_windows()

        except KeyboardInterrupt:
            print("\n🛑 Shutting down system...")
        finally:
            self.stop_event.set()
            self.rtmp_hub.stop_all()
            if self.display_thread is not None and self.display_thread.is_alive():
                self.display_thread.join(timeout=1.0)
            cv2.destroyAllWindows()

    def setup(self):
        start_funnel()

        ingestion_service = get_ingestion_service()
        self.camera_ids = list(ingestion_service.camera_config.keys())

        self.backbone = SharedBackbone()

        # Only enable local OpenCV preview when in development *and* RTMP publishing is disabled.
        if self.env == "development" and not ENABLE_RTMP_PUBLISH:
            self.display_frames = {cam_id: None for cam_id in self.camera_ids}
            print("[Display] Running display rendering on main runtime thread.")

        if ENABLE_RTMP_PUBLISH:
            if ffmpeg_exists(RTMP_FFMPEG_BIN):
                print(f"[RTMP] Publisher enabled. Base URL: {RTMP_BASE_URL}")
            else:
                print(
                    f"[RTMP] WARNING: ffmpeg binary '{RTMP_FFMPEG_BIN}' not found. "
                    "RTMP publishing is enabled but will fail until ffmpeg is installed/available."
                )

        print("🔌 Loading AI Plugins (controlled by camera_config.detections)...")
        self.plugins = []

        # Determine which cameras have detections enabled and pass that list
        # to plugins that accept per-camera configuration.
        detection_enabled_cams = [cam for cam in self.camera_ids if self._is_detection_enabled(cam)]

        try:
            self.plugins.append(TusslePlugin(model_path=TUSSLE_MODEL_PATH, camera_ids=detection_enabled_cams))
        except Exception as exc:
            print(f"⚠️ [Plugins] Failed to initialize TusslePlugin: {exc}")

        try:
            # Rule-based fall plugin processes the scene_state passed by runtime,
            # and runtime will filter out cameras without detections before calling plugins.
            self.plugins.append(RuleBasedFallPlugin())
        except Exception as exc:
            print(f"⚠️ [Plugins] Failed to initialize RuleBasedFallPlugin: {exc}")

        self.recorder = ClipRecorder(
            before_frames=CLIP_RECORDER_BEFORE_FRAMES,
            after_frames=CLIP_RECORDER_AFTER_FRAMES,
            fps=CLIP_RECORDER_FPS,
        )
        print("🚀 Master Loop Online. Analyzing streams...")

