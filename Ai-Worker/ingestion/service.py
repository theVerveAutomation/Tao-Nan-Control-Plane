import json
import os
import queue
import select
import threading
import time

import cv2

from .repository import PostgresCameraRepository


class IngestionService:
    def __init__(self, camera_config=None, env=None, repository=None, db_available=True):
        self.camera_config = camera_config or {
            "cam1": {
                "url": "./Tussle.mp4",
                "is_active": True,
                "fall_detection": True,
                "tussle_detection": True,
            }
        }
        self.env = (env or os.environ.get("ENV", "development")).lower()
        self.db_available = db_available

        self.state_lock = threading.Lock()
        self.camera_threads = {}
        self.frame_queues = {cam_id: queue.Queue(maxsize=2) for cam_id in self.camera_config}

        self.repository = repository if repository is not None else PostgresCameraRepository()

    def bootstrap_cameras_from_db(self):
        if not self.db_available:
            return

        cameras_config = self.repository.fetch_all_cameras()
        if not cameras_config:
            print("⚠️ [Bootstrap] No cameras found in DB. Using local fallback config.")
            return

        with self.state_lock:
            self.camera_config.clear()
            self.camera_config.update(cameras_config)
            self.frame_queues.clear()
            for cam_id in self.camera_config:
                self.frame_queues[cam_id] = queue.Queue(maxsize=2)

        print(f"✅ [Bootstrap] Loaded {len(self.camera_config)} cameras from DB.")

    def apply_single_camera_update_from_db(self, cam_id):
        camera_row = self.repository.fetch_camera_by_id(cam_id)
        if not camera_row:
            print(f"⚠️ [Realtime] Camera {cam_id} not found in DB (possibly deleted).")
            return None

        source_url = camera_row.get("stream_url") or camera_row.get("url")
        if not source_url:
            print(f"⚠️ [Realtime] Camera {cam_id} has no stream URL. Skipping stream update.")

        with self.state_lock:
            db_camera = self.repository.camera_from_db_row(camera_row)
            if cam_id not in self.camera_config:
                self.camera_config[cam_id] = db_camera
            else:
                self.camera_config[cam_id].update(db_camera)

            if cam_id not in self.frame_queues:
                self.frame_queues[cam_id] = queue.Queue(maxsize=2)

        return camera_row

    def listen_to_postgres(self):
        if not self.db_available:
            return

        try:
            conn = self.repository.get_connection()
            curs = conn.cursor()
            curs.execute("LISTEN camera_update_channel;")
            print("✅ [Realtime] Connected to PostgreSQL. Listening for config updates...")

            while True:
                if select.select([conn], [], [], 5) == ([], [], []):
                    continue

                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    try:
                        payload = json.loads(notify.payload)
                    except Exception:
                        payload = notify.payload

                    print(f"🔔 [Realtime] Received DB notification: {payload}")
                    if isinstance(payload, dict):
                        cam_id = payload.get("id") or payload.get("cameraId")
                    else:
                        cam_id = str(payload)

                    if not cam_id:
                        continue

                    print(f"🔄 [Realtime] Fetching latest DB row for {cam_id}...")
                    updated_row = self.apply_single_camera_update_from_db(cam_id)
                    if updated_row is None:
                        continue

                    with self.state_lock:
                        cam_cfg = self.camera_config.get(cam_id, {})
                        is_active = cam_cfg.get("is_active", False)
                        stream_url = cam_cfg.get("url")

                    if is_active and stream_url and cam_id not in self.camera_threads:
                        t = threading.Thread(target=self.capture_stream, args=(cam_id, stream_url), daemon=True)
                        t.start()
                        self.camera_threads[cam_id] = t
                        print(f"✅ [Realtime] Started stream thread for {cam_id}")
        except Exception as exc:
            print(f"❌ [Realtime] Database listener failed: {exc}. Running on default config.")

    def capture_stream(self, cam_id, stream_url):
        reconnect_delay = 2
        max_read_failures = 30
        dev_target_fps = 30.0
        dev_frame_interval = 1.0 / dev_target_fps

        while True:
            print(f"[{cam_id}] Connecting to stream...")
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"[{cam_id}] ❌ ERROR: Could not open source.")
                return

            consecutive_failures = 0

            while cap.isOpened():
                frame_start_time = time.time()
                ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures < max_read_failures:
                        time.sleep(0.02)
                        continue
                    print(f"[{cam_id}] ⚠️ {max_read_failures} consecutive failures. Reconnecting...")
                    break

                consecutive_failures = 0

                with self.state_lock:
                    cam_queue = self.frame_queues.get(cam_id)

                if cam_queue is None:
                    time.sleep(0.01)
                    continue

                if cam_queue.full():
                    try:
                        cam_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    cam_queue.put_nowait(frame)
                except queue.Full:
                    pass

                if self.env == "development":
                    elapsed = time.time() - frame_start_time
                    sleep_time = dev_frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            cap.release()
            print(f"[{cam_id}] ⏳ Retrying in {reconnect_delay}s...")
            time.sleep(reconnect_delay)

    def start(self):
        print("🚀 Starting ingestion funnel...")

        try:
            self.bootstrap_cameras_from_db()
        except Exception as exc:
            print(f"⚠️ [Bootstrap] Failed to load cameras from DB: {exc}. Using fallback config.")

        db_thread = threading.Thread(target=self.listen_to_postgres, daemon=True)
        db_thread.start()

        threads = []
        with self.state_lock:
            cameras_snapshot = dict(self.camera_config)

        for cam_id, cam_cfg in cameras_snapshot.items():
            stream_url = cam_cfg.get("url")
            if not stream_url:
                continue
            t = threading.Thread(target=self.capture_stream, args=(cam_id, stream_url), daemon=True)
            t.start()
            threads.append(t)
            self.camera_threads[cam_id] = t

        print("✅ Funnel is active and streaming.")
        return threads
