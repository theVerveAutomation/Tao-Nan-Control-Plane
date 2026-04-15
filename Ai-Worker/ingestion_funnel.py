import os
import time

from ingestion.recorder import EventRecorder
from ingestion.service import IngestionService

try:
    import psycopg2  # noqa: F401
    DB_AVAILABLE = True
except ImportError:
    print("⚠️ psycopg2 not installed. Database hot-reloading disabled.")
    DB_AVAILABLE = False

ENV = os.environ.get("ENV", "development").lower()
print(f"⚙️  Environment set to: {ENV.upper()}")

# Singleton service used across the app
_ingestion_service = IngestionService(env=ENV, db_available=DB_AVAILABLE)

# Compatibility exports (worker.py imports these names directly)
CAMERA_CONFIG = _ingestion_service.camera_config
frame_queues = _ingestion_service.frame_queues
state_lock = _ingestion_service.state_lock
camera_threads = _ingestion_service.camera_threads


def start_funnel():
    return _ingestion_service.start()


def get_ingestion_service():
    return _ingestion_service


if __name__ == "__main__":
    start_funnel()

    with state_lock:
        recorders = {cam_id: EventRecorder(cam_id) for cam_id in CAMERA_CONFIG}

    print("🚀 Starting AI processing loop. Press Ctrl+C to stop.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            current_batch = {}

            with state_lock:
                queue_items = list(frame_queues.items())

            for cam_id, q in queue_items:
                if not q.empty():
                    frame = q.get()
                    current_batch[cam_id] = frame

                    if cam_id not in recorders:
                        recorders[cam_id] = EventRecorder(cam_id)
                    recorders[cam_id].update_buffer(frame)

            if not current_batch:
                time.sleep(0.01)
                continue

            active_cams = 0
            for cam_id in current_batch:
                config = CAMERA_CONFIG.get(cam_id, {})
                if not config.get("is_active", True):
                    continue

                active_cams += 1
                if config.get("fall_detection", True):
                    pass

            time.sleep(0.1)

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                with state_lock:
                    total_cams = len(CAMERA_CONFIG)
                print(f"⚡ Processing {active_cams}/{total_cams} active cameras | AI Pipeline Speed: {fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\n🛑 Shutting down ingestion funnel...")
        for recorder in recorders.values():
            if recorder.is_recording:
                recorder.stop_recording()
