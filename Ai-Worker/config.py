import os
from pathlib import Path
from dotenv import load_dotenv
from copy import deepcopy

# Load local .env (Ai-Worker/.env) so environment variables are available to config
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


ENV = os.environ.get("ENV", "development").lower()
DATABASE_URL = os.environ.get("DATABASE_URL")
ALERT_CREATE_URL = os.environ.get("ALERT_CREATE_URL", "http://localhost:5000/api/alerts")

DEFAULT_CAMERA_CONFIG = {
    "cam1": {
        "url": "./Tussle.mp4",
        "is_active": True,
        "fall_detection": True,
        "tussle_detection": True,
    }
}

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n-pose.pt")
TUSSLE_MODEL_PATH = os.environ.get("TUSSLE_MODEL_PATH", "./best_slowfast_fight_model (2).pth")

ENABLE_TUSSLE_PLUGIN = os.environ.get("ENABLE_TUSSLE_PLUGIN", "false" \
"").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_FALL_PLUGIN = os.environ.get("ENABLE_FALL_PLUGIN", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

ALERT_COOLDOWN_SECONDS = float(os.environ.get("ALERT_COOLDOWN_SECONDS", "5"))

CLIP_RECORDER_BEFORE_FRAMES = int(os.environ.get("CLIP_RECORDER_BEFORE_FRAMES", "200"))
CLIP_RECORDER_AFTER_FRAMES = int(os.environ.get("CLIP_RECORDER_AFTER_FRAMES", "200"))
CLIP_RECORDER_FPS = int(os.environ.get("CLIP_RECORDER_FPS", "10"))

FRAME_QUEUE_MAXSIZE = int(os.environ.get("FRAME_QUEUE_MAXSIZE", "2"))
STREAM_RECONNECT_DELAY_SECONDS = float(os.environ.get("STREAM_RECONNECT_DELAY_SECONDS", "2"))
STREAM_MAX_READ_FAILURES = int(os.environ.get("STREAM_MAX_READ_FAILURES", "30"))
DEV_TARGET_FPS = float(os.environ.get("DEV_TARGET_FPS", "30"))

ENABLE_RTMP_PUBLISH = os.environ.get("ENABLE_RTMP_PUBLISH", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RTMP_BASE_URL_DETECTION = os.environ.get("RTMP_BASE_URL_DETECTION", "rtmp://localhost:1935/detection")
RTMP_BASE_URL_LIVE = os.environ.get("RTMP_BASE_URL_LIVE", "rtmp://localhost:1935/live")
RTMP_STREAM_PREFIX = os.environ.get("RTMP_STREAM_PREFIX", "cam_")
RTMP_PUBLISH_FPS = int(os.environ.get("RTMP_PUBLISH_FPS", "10"))
RTMP_FRAME_WIDTH = int(os.environ.get("RTMP_FRAME_WIDTH", "640"))
RTMP_FRAME_HEIGHT = int(os.environ.get("RTMP_FRAME_HEIGHT", "360"))
RTMP_FFMPEG_BIN = os.environ.get("RTMP_FFMPEG_BIN", "ffmpeg")
RTMP_VIDEO_CODEC = os.environ.get("RTMP_VIDEO_CODEC", "libx264")
RTMP_FFMPEG_PRESET = os.environ.get("RTMP_FFMPEG_PRESET", "veryfast")

# MediaMTX REST API base URL (paths endpoint)
MEDIAMTX_BASE_URL = os.environ.get("MEDIAMTX_BASE_URL", "http://localhost:9997/v3/paths/list")


def get_default_camera_config():
    return deepcopy(DEFAULT_CAMERA_CONFIG)
