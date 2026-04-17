import json
import queue
import re
import select
import threading
import time

import cv2
import requests

from config import (
    DEV_TARGET_FPS,
    ENV,
    FRAME_QUEUE_MAXSIZE,
    STREAM_MAX_READ_FAILURES,
    STREAM_RECONNECT_DELAY_SECONDS,
    get_default_camera_config,
    MEDIAMTX_BASE_URL,
)
from .repository import PostgresCameraRepository


class IngestionService:
    def __init__(self, camera_config=None, env=None, repository=None, db_available=True):
        self.camera_config = camera_config or get_default_camera_config()
        self.env = (env or ENV).lower()
        self.db_available = db_available

        self.state_lock = threading.Lock()
        self.camera_threads = {}
        self.stop_events = {}
        self.thread_stream_url = {}
        self.frame_queues = {cam_id: queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE) for cam_id in self.camera_config}

        self.repository = repository if repository is not None else PostgresCameraRepository()
        self.mediamtx_base_url = MEDIAMTX_BASE_URL

    @staticmethod
    def _mediamtx_path_name(cam_id):
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(cam_id))
        return f"cam_{safe_id}"

    def _upsert_mediamtx_path(self, cam_id, camera_row):
        if not cam_id or not camera_row:
            return

        source_url = camera_row.get("stream_url") or camera_row.get("url")
        if not source_url:
            print(f"⚠️ [MediaMTX] Camera {cam_id} has no source URL. Skipping path upsert.")
            return

        path_name = self._mediamtx_path_name(cam_id)
        payload = {
            "name": path_name,
            "source": source_url,
            "sourceOnDemand": True,
        }

        # Primary call follows requested endpoint shape; fallback supports path-name URL styles.
        try:
            response = requests.post(self.mediamtx_base_url, json=payload, timeout=3)
            if response.status_code not in (200, 201, 204):
                alt_url = f"{self.mediamtx_base_url}/{path_name}"
                alt_payload = {
                    "source": source_url,
                    "sourceOnDemand": True,
                }
                alt_response = requests.post(alt_url, json=alt_payload, timeout=3)
                if alt_response.status_code not in (200, 201, 204):
                    print(
                        f"⚠️ [MediaMTX] Failed to upsert path {path_name}. "
                        f"POST {self.mediamtx_base_url} -> {response.status_code}; "
                        f"POST {alt_url} -> {alt_response.status_code}"
                    )
                    return
            print(f"✅ [MediaMTX] Upserted path {path_name} (sourceOnDemand=true)")
        except Exception as exc:
            print(f"❌ [MediaMTX] Upsert failed for camera {cam_id}: {exc}")

    def _delete_mediamtx_path(self, cam_id):
        if not cam_id:
            return

        path_name = self._mediamtx_path_name(cam_id)
        target_url = f"{self.mediamtx_base_url}/{path_name}"

        try:
            response = requests.delete(target_url, timeout=3)
            if response.status_code in (200, 202, 204, 404):
                print(f"✅ [MediaMTX] Deleted path {path_name}")
                return

            # Fallback for APIs expecting payload on collection endpoint
            fallback_response = requests.delete(self.mediamtx_base_url, json={"name": path_name}, timeout=3)
            if fallback_response.status_code in (200, 202, 204, 404):
                print(f"✅ [MediaMTX] Deleted path {path_name}")
            else:
                print(
                    f"⚠️ [MediaMTX] Failed to delete path {path_name}. "
                    f"DELETE {target_url} -> {response.status_code}; "
                    f"DELETE {self.mediamtx_base_url} -> {fallback_response.status_code}"
                )
        except Exception as exc:
            print(f"❌ [MediaMTX] Delete failed for camera {cam_id}: {exc}")

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
                self.frame_queues[cam_id] = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

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
                self.frame_queues[cam_id] = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

        return camera_row

    def _parse_notify_payload(self, notify):
        try:
            payload = json.loads(notify.payload)
        except Exception:
            payload = notify.payload

        op = None
        if isinstance(payload, dict):
            # support explicit operation key if provided by trigger
            for key in ("operation", "op", "action", "_op", "tg_op"):
                if key in payload:
                    try:
                        op = str(payload.get(key)).lower()
                    except Exception:
                        op = None
                    break

            cam_id = payload.get("data").get("id")
        else:
            cam_id = str(payload)

        return cam_id, payload, op

    def _get_camera_row(self, cam_id, payload):
        if isinstance(payload, dict):
            return payload
        return self.repository.fetch_camera_by_id(cam_id)

    def _handle_delete(self, cam_id):
        with self.state_lock:
            # signal running thread to stop
            ev = self.stop_events.get(cam_id)
            if ev is not None:
                try:
                    ev.set()
                except Exception:
                    pass

            # remove runtime state
            self.camera_config.pop(cam_id, None)
            self.frame_queues.pop(cam_id, None)

            # join and remove thread reference
            thread = self.camera_threads.pop(cam_id, None)
            # remove stored stream URL for this camera (avoid stale entries)
            self.thread_stream_url.pop(cam_id, None)

        if thread is not None:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

        with self.state_lock:
            self.stop_events.pop(cam_id, None)

    def _update_camera_config_from_row(self, cam_id, camera_row):
        if not camera_row:
            return False

        source_url = camera_row.get("url")
        if not source_url:
            print(f"⚠️ [Realtime] Camera {cam_id} has no stream URL. Skipping stream update.")

        with self.state_lock:
            db_camera = self.repository.camera_from_db_row(camera_row)
            if cam_id not in self.camera_config:
                self.camera_config[cam_id] = db_camera
            else:
                self.camera_config[cam_id].update(db_camera)

            if cam_id not in self.frame_queues:
                self.frame_queues[cam_id] = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

        return True

    def _ensure_stream_thread(self, cam_id):
        with self.state_lock:
            cam_cfg = self.camera_config.get(cam_id, {})
            detection = cam_cfg.get("detection", False)
            stream_url = cam_cfg.get("url")

            print(f"✅ [Realtime] Ensuring stream thread for {cam_id}: {stream_url} - Detection: {detection}")
            print(f"✅ [Realtime] Current threads: {list(self.camera_threads.keys())}")


        if not detection or not stream_url:
            return


        existing_thread = self.camera_threads.get(cam_id)
        current_url = self.thread_stream_url.get(cam_id)

        # If there's a running thread
        if existing_thread is not None and existing_thread.is_alive():
            print(f"✅ [Realtime] Existing thread found for {cam_id}")
            # If the URL hasn't changed, nothing to do
            if current_url == stream_url:
                return

            # URL changed: request stop and replace the thread
            print(f"🔁 [Realtime] Stream URL changed for {cam_id}, restarting thread: {current_url} -> {stream_url}")
            ev = self.stop_events.get(cam_id)
            if ev:
                ev.set()
            try:
                existing_thread.join(timeout=1.0)
            except Exception:
                pass
            # cleanup references
            self.camera_threads.pop(cam_id, None)
            self.stop_events.pop(cam_id, None)
            self.thread_stream_url.pop(cam_id, None)

        # If no live thread exists, start one
        if cam_id not in self.camera_threads:
            print(f"✅ [Realtime] Creating stream thread for {cam_id}: {stream_url}")
            ev = threading.Event()
            self.stop_events[cam_id] = ev

            t = threading.Thread(target=self.capture_stream, args=(cam_id, stream_url, ev), daemon=True)
            t.start()
            self.camera_threads[cam_id] = t
            self.thread_stream_url[cam_id] = stream_url
            print(f"✅ [Realtime] Started stream thread for {cam_id}")

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
                        cam_id, payload, op = self._parse_notify_payload(notify)

                        print(f"🔔 [Realtime] Received DB notification: {payload}")

                        if not cam_id:
                            continue

                        # explicit delete operation in payload
                        if op and op.startswith("delete"):
                            print(f"🔔 [Realtime] Handling delete operation for camera {cam_id}")
                            self._delete_mediamtx_path(cam_id)
                            self._handle_delete(cam_id)
                            continue

                        # Obtain camera row from payload or DB (fallback)
                        if isinstance(payload, dict):
                            camera_row = payload.get('data')
                        else:
                            camera_row = self.repository.fetch_camera_by_id(cam_id)
                            if not camera_row:
                                self._handle_delete(cam_id)
                                continue

                        # Update in-memory state and ensure a stream thread exists
                        self._update_camera_config_from_row(cam_id, camera_row)
                        self._upsert_mediamtx_path(cam_id, camera_row)
                        self._ensure_stream_thread(cam_id)
        except Exception as exc:
            print(f"❌ [Realtime] Database listener failed: {exc}. Running on default config.")

    def capture_stream(self, cam_id, stream_url, stop_event):
        reconnect_delay = STREAM_RECONNECT_DELAY_SECONDS
        max_read_failures = STREAM_MAX_READ_FAILURES
        dev_target_fps = DEV_TARGET_FPS
        dev_frame_interval = 1.0 / dev_target_fps

        while True:
            # stop event requested?
            if stop_event.is_set():
                print(f"[{cam_id}] ⛔ Stop requested. Exiting capture loop.")
                return
            print(f"[{cam_id}] Connecting to stream...")
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"[{cam_id}] ❌ ERROR: Could not open source.")
                return
            else:
                print(f"[{cam_id}] ✅ Opened source: {stream_url}")

            consecutive_failures = 0

            while cap.isOpened():
                # check stop event frequently so we can exit quickly when asked
                if stop_event.is_set():
                    print(f"[{cam_id}] ⛔ Stop requested while reading. Releasing capture and exiting.")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    return

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
                    # drop oldest frame to make room
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
            # create stop event for this camera thread
            ev = threading.Event()
            self.stop_events[cam_id] = ev

            t = threading.Thread(target=self.capture_stream, args=(cam_id, stream_url, ev), daemon=True)
            t.start()
            threads.append(t)
            self.camera_threads[cam_id] = t
            self.thread_stream_url[cam_id] = stream_url

        print("✅ Funnel is active and streaming.")
        return threads
