import time

import cv2


def display_loop(stop_event, display_frames, display_lock):
    """Dedicated display loop so rendering does not block AI processing."""
    while not stop_event.is_set():
        has_frame = False
        with display_lock:
            items = list(display_frames.items())

        for cam_id, frame in items:
            if frame is None:
                continue
            has_frame = True
            cv2.imshow(f"Tracked Output {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Display] Window closed by user.")
            stop_event.set()
            break

        if not has_frame:
            time.sleep(0.005)

    cv2.destroyAllWindows()
