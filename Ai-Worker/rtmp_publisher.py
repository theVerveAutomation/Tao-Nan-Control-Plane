import re
import subprocess
import threading
import time
from typing import Dict, Optional

import cv2


def _to_safe_stream_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", str(value))


class FFmpegRtmpPublisher:
    def __init__(
        self,
        camera_id: str,
        output_url: str,
        fps: int = 10,
        width: int = 640,
        height: int = 360,
        ffmpeg_bin: str = "ffmpeg",
        video_codec: str = "libx264",
        preset: str = "veryfast",
    ):
        self.camera_id = str(camera_id)
        self.output_url = output_url
        self.fps = int(max(1, fps))
        self.width = int(max(1, width))
        self.height = int(max(1, height))
        self.ffmpeg_bin = ffmpeg_bin
        self.video_codec = video_codec
        self.preset = preset
        self._min_publish_interval = 1.0 / float(self.fps)
        self._last_publish_ts = 0.0

        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _build_command(self):
        # Raw BGR frames from stdin -> H264 -> RTMP/FLV
        return [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            self.video_codec,
            "-preset",
            self.preset,
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "flv",
            self.output_url,
        ]

    def _ensure_started(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True

        command = self._build_command()
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[RTMP] Started ffmpeg publisher for {self.camera_id} -> {self.output_url}")
            return True
        except Exception as exc:
            print(f"[RTMP] Failed to start ffmpeg for {self.camera_id}: {exc}")
            self._process = None
            return False

    def publish(self, frame):
        with self._lock:
            if not self._ensure_started():
                return

            if frame is None:
                return

            now = time.time()
            if now - self._last_publish_ts < self._min_publish_interval:
                return
            self._last_publish_ts = now

            try:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                if self._process is None or self._process.stdin is None:
                    return

                self._process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                print(f"[RTMP] Publisher pipe error for {self.camera_id}: {exc}. Restarting next frame.")
                self.stop()
            except Exception as exc:
                print(f"[RTMP] Failed to publish frame for {self.camera_id}: {exc}")

    def stop(self):
        with self._lock:
            if self._process is None:
                return

            proc = self._process
            self._process = None

            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass

            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

            print(f"[RTMP] Stopped publisher for {self.camera_id}")


class RtmpPublisherHub:
    def __init__(
        self,
        enabled: bool,
        base_url: str,
        stream_prefix: str,
        fps: int,
        width: int,
        height: int,
        ffmpeg_bin: str,
        video_codec: str,
        preset: str,
    ):
        self.enabled = bool(enabled)
        self.base_url = base_url.rstrip("/")
        self.stream_prefix = stream_prefix
        self.fps = int(max(1, fps))
        self.width = int(max(1, width))
        self.height = int(max(1, height))
        self.ffmpeg_bin = ffmpeg_bin
        self.video_codec = video_codec
        self.preset = preset

        self._publishers: Dict[str, FFmpegRtmpPublisher] = {}
        self._lock = threading.Lock()

    def _output_url(self, camera_id: str) -> str:
        safe_id = _to_safe_stream_key(camera_id)
        stream_key = f"{self.stream_prefix}{safe_id}"
        return f"{self.base_url}/{stream_key}"

    def publish_frame(self, camera_id: str, frame):
        if not self.enabled:
            return

        cam_id = str(camera_id)
        with self._lock:
            publisher = self._publishers.get(cam_id)
            if publisher is None:
                publisher = FFmpegRtmpPublisher(
                    camera_id=cam_id,
                    output_url=self._output_url(cam_id),
                    fps=self.fps,
                    width=self.width,
                    height=self.height,
                    ffmpeg_bin=self.ffmpeg_bin,
                    video_codec=self.video_codec,
                    preset=self.preset,
                )
                self._publishers[cam_id] = publisher

        publisher.publish(frame)

    def prune_removed_cameras(self, active_camera_ids):
        if not self.enabled:
            return

        active = {str(cam_id) for cam_id in active_camera_ids}
        stale = []

        with self._lock:
            for cam_id in list(self._publishers.keys()):
                if cam_id not in active:
                    stale.append((cam_id, self._publishers.pop(cam_id)))

        for cam_id, publisher in stale:
            publisher.stop()
            print(f"[RTMP] Removed stale publisher for camera {cam_id}")

    def stop_all(self):
        with self._lock:
            items = list(self._publishers.items())
            self._publishers.clear()

        for _, publisher in items:
            publisher.stop()


def ffmpeg_exists(ffmpeg_bin: str = "ffmpeg") -> bool:
    try:
        subprocess.run(
            [ffmpeg_bin, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception:
        return False
