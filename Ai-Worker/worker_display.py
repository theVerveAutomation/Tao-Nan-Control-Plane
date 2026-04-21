import time
import cv2
import threading


def display_loop(stop_event, display_frames, display_lock):
    """Dedicated display loop so rendering does not block AI processing."""
    opened_windows = set()
    frame_publish_count = {}
    last_health_log_at = 0.0
    loop_count = 0
    # lazy import to avoid circular import at module load time
    CAMERA_CONFIG = None
    print(f"[Display] Loop started on thread: {threading.current_thread().name}")

    while not stop_event.is_set():
        loop_count += 1
        has_frame = False
        with display_lock:
            items = list(display_frames.items())

        if loop_count % 200 == 0:
            ids_snapshot = [cam_id for cam_id, _ in items]
            ready_snapshot = [cam_id for cam_id, frame in items if frame is not None]
            print(
                f"[Display][Debug] frame keys={ids_snapshot} | ready={ready_snapshot} | opened={list(opened_windows)}"
            )

        # lazy-resolve CAMERA_CONFIG; retry on failure instead of freezing to {}
        if CAMERA_CONFIG is None:
            try:
                from ingestion_funnel import CAMERA_CONFIG as _CAM
                CAMERA_CONFIG = _CAM
            except Exception:
                CAMERA_CONFIG = None

        for cam_id, frame in items:
            if frame is None:
                continue
            has_frame = True
            window_name = f"Tracked Output {cam_id}"

            count = frame_publish_count.get(cam_id, 0) + 1
            frame_publish_count[cam_id] = count
            if count <= 3:
                try:
                    print(
                        f"[Display][Debug] First frames for {cam_id}: count={count}, shape={frame.shape}, dtype={frame.dtype}"
                    )
                except Exception:
                    print(f"[Display][Debug] First frames for {cam_id}: count={count}")

            if cam_id not in opened_windows:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    opened_windows.add(cam_id)
                    prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                    print(f"[Display] Opening window for camera {cam_id} | visible_prop={prop}")
                except Exception as exc:
                    print(f"[Display] Failed to create window for camera {cam_id}: {exc}")
                    continue

            try:
                cv2.imshow(window_name, frame)
            except Exception as exc:
                print(f"[Display] Failed to render frame for camera {cam_id}: {exc}")
                opened_windows.discard(cam_id)
                continue

            if count <= 3:
                try:
                    prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                    print(f"[Display][Debug] Post-imshow window state for {cam_id}: visible_prop={prop}")
                except Exception as exc:
                    print(f"[Display][Debug] Could not query window state for {cam_id}: {exc}")

        # Close windows for cameras removed from runtime state.
        # Prefer CAMERA_CONFIG when available; fall back to current display_frames keys.
        try:
            if isinstance(CAMERA_CONFIG, dict):
                current_ids = set(CAMERA_CONFIG.keys())
            else:
                current_ids = {cam_id for cam_id, _ in items}
        except Exception:
            current_ids = {cam_id for cam_id, _ in items}

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

        now = time.time()
        if now - last_health_log_at >= 2.0:
            print(
                f"[Display][Health] opened={list(opened_windows)} | total_keys={len(items)} | ready={sum(1 for _, f in items if f is not None)}"
            )
            last_health_log_at = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Display] Window closed by user.")
            stop_event.set()
            break

        if not has_frame:
            time.sleep(0.005)

    cv2.destroyAllWindows()
