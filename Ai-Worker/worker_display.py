import time
import cv2


def display_loop(stop_event, display_frames, display_lock):
    """Dedicated display loop so rendering does not block AI processing."""
    opened_windows = set()
    # lazy import to avoid circular import at module load time
    CAMERA_CONFIG = None
    while not stop_event.is_set():
        has_frame = False
        with display_lock:
            items = list(display_frames.items())

        # lazy-resolve CAMERA_CONFIG once
        if CAMERA_CONFIG is None:
            try:
                from ingestion_funnel import CAMERA_CONFIG as _CAM
                CAMERA_CONFIG = _CAM
            except Exception:
                CAMERA_CONFIG = {}

        for cam_id, frame in items:
            if frame is None:
                continue
            has_frame = True
            if cam_id not in opened_windows:
                try:
                    cv2.namedWindow(f"Tracked Output {cam_id}", cv2.WINDOW_NORMAL)
                except Exception:
                    pass
                opened_windows.add(cam_id)
                print(f"[Display] Opening window for camera {cam_id}")

            cv2.imshow(f"Tracked Output {cam_id}", frame)

        # Close windows for cameras removed from CAMERA_CONFIG
        try:
            current_ids = set(CAMERA_CONFIG.keys()) if isinstance(CAMERA_CONFIG, dict) else set()
        except Exception:
            current_ids = set()

        to_close = set(opened_windows) - current_ids
        if to_close:
            with display_lock:
                for cid in to_close:
                    try:
                        cv2.destroyWindow(f"Tracked Output {cid}")
                    except Exception:
                        pass
                    opened_windows.discard(cid)
                    # remove stale frame buffer if present
                    display_frames.pop(cid, None)
                    print(f"[Display] Closed window for removed camera {cid}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Display] Window closed by user.")
            stop_event.set()
            break

        if not has_frame:
            time.sleep(0.005)

    cv2.destroyAllWindows()
