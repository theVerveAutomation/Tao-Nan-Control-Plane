import cv2
import threading
import queue
import time
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
CAMERAS = [
    {"id": "cam1", "url": "./Tussle.mp4"},
]

ENV = os.environ.get("ENV", "development").lower()
print(f"⚙️  Environment set to: {ENV.upper()}")

# ==========================================
# 2. THE QUEUES (The "Drop-Oldest" Buffer)
# ==========================================
# maxsize=2 
# It holds the current frame being read, and the absolute newest frame waiting for the AI.
frame_queues = {cam["id"]: queue.Queue(maxsize=2) for cam in CAMERAS}

# ==========================================
# 3. THE PRODUCER (Background Camera Threads)
# ==========================================
def capture_stream(cam_id, stream_url):
    """
    Background thread for a specific camera.
    Now upgraded with your VLC-style robust reconnection logic!
    """
    RECONNECT_DELAY = 2        # seconds to wait before rebooting connection
    MAX_READ_FAILURES = 30     # consecutive None frames before forcing a reconnect
    DEV_TARGET_FPS = 30.0
    DEV_FRAME_INTERVAL = 1.0 / DEV_TARGET_FPS
    
    while True: # Outer loop: Reconnection Engine
        print(f"[{cam_id}] Connecting to stream...")
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[{cam_id}] ❌ ERROR: Could not open video source '{stream_url}'. Check file path, permissions, or codec support.")
            return  # Exit the thread if the file cannot be opened

        consecutive_failures = 0

        while cap.isOpened(): # Inner loop: Real-time Grabbing
            frame_start_time = time.time()
            ret, frame = cap.read()

            # --- YOUR FAULT TOLERANCE LOGIC ---
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures < MAX_READ_FAILURES:
                    time.sleep(0.02)  # brief wait for micro-stutter, then try again
                    continue
                
                print(f"[{cam_id}] ⚠️ {MAX_READ_FAILURES} consecutive read failures. Forcing reconnect...")
                break # Break the inner loop to trigger the outer reconnection loop

            # If we successfully read a frame, reset the failure counter
            consecutive_failures = 0
            
            # --- OUR DROP-OLDEST QUEUE LOGIC ---
            if frame_queues[cam_id].full():
                try:
                    print(f"[{cam_id}] Queue full. Dropping oldest frame to make room for new one.")
                    frame_queues[cam_id].get_nowait()
                except queue.Empty:
                    pass
            
            try:
                frame_queues[cam_id].put_nowait(frame)
            except queue.Full:
                pass

            # Enforce source read pace at 30 FPS in development mode
            if ENV == "development":
                elapsed = time.time() - frame_start_time
                sleep_time = DEV_FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # If we break out of the inner loop, release the camera and wait before retrying
        cap.release()
        print(f"[{cam_id}] ⏳ Retrying in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)
# ==========================================
# 4. INITIALIZATION
# ==========================================
def start_funnel():
    """Spins up all background threads safely."""
    print("🚀 Starting ingestion funnel...")
    threads = []
    for cam in CAMERAS:
        # daemon=True means these threads will automatically die when the main script stops
        t = threading.Thread(target=capture_stream, args=(cam["id"], cam["url"]), daemon=True)
        t.start()
        threads.append(t)
    
    # Give cameras a second to warm up and fill the queues
    print("⏳ Warming up camera streams...")
    #time.sleep(2)
    print("✅ Funnel is active and streaming.")
    return threads

# ==========================================
# 5. THE CONSUMER (Your Main AI Loop)
# ==========================================
if __name__ == "__main__":
    
    start_funnel()
    
    print("🚀 Starting AI processing loop. Press Ctrl+C to stop.")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 1. Prepare the batch container
            current_batch = {}
            
            # 2. Extract the absolute newest frame from every camera
            for cam_id, q in frame_queues.items():
                if not q.empty():
                    current_batch[cam_id] = q.get()
            
            # 3. Verify we have data before running heavy AI
            if not current_batch:
                time.sleep(0.01) # Sleep 10ms to prevent CPU maxing out while waiting
                continue
            
            # --------------------------------------------------
            # 🧠 YOUR AI GOES HERE
            # e.g., results = yolo_model(list(current_batch.values()))
            # --------------------------------------------------
            
            # For this simulation, we simulate heavy AI compute taking 0.1 seconds (10 FPS max)
            time.sleep(0.1) 
            
            # Calculate and print FPS to prove it is working
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                active_cams = len(current_batch)
                print(f"⚡ Processing {active_cams}/4 cameras | AI Pipeline Speed: {fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\n🛑 Shutting down ingestion funnel...")
        cv2.destroyAllWindows()