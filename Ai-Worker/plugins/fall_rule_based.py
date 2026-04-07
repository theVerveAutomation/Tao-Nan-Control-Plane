import time
from collections import defaultdict, deque

import numpy as np


class RuleBasedFallPlugin:
    """
    Advanced fall detector adapted from your standalone script, integrated for
    the existing plugin interface: process_batch(scene_state) -> alerts.
    """

    def __init__(
        self,
        confidence_threshold=0.70,
        aspect_ratio_threshold=1.2,
        overhead_camera_override=None,
        debug=False,
    ):
        print("⚡ Initializing Fall Detection Plugin (Advanced Rule-Based)...")

        # Keep legacy args for compatibility with existing worker wiring.
        self.conf_threshold = confidence_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold

        self.debug = debug
        self.overhead_camera_override = overhead_camera_override

        # ---- Core config (from provided logic, plugin-adapted) ----
        self.BASELINE_TIME = 1.5
        self.UPRIGHT_ANGLE = 18
        self.GROUND_CONFIRM_TIME = 0.10
        self.MIN_DESCENT_FRAMES = 2
        self.RECOVERY_COOLDOWN = 5.0
        self.INERTIA_TIME = 0.6
        self.KP_CONF_THRESH = 0.30

        self.DESCENT_VEL_THRESH = 120.0
        self.IMPACT_VEL_THRESH = 1000.0
        self.IMPACT_VEL_CAP_COLD = 5000.0
        self.IMPACT_VEL_CAP_HOT = 20000.0
        self.HOT_DESCENT_FRAMES = 3

        self.SMOOTH_WINDOW = 5
        self.CRITICAL_KPS = [5, 6, 11, 12, 13, 14, 15, 16]

        # Camera auto-mode sampling from incoming detections.
        self.AUTO_SAMPLE_DETECTIONS = 80
        self.mode_ratio_samples = defaultdict(list)
        self.mode_angle_samples = defaultdict(list)
        self.camera_mode = {}  # cam_id -> bool (True=overhead, False=side)

        # ---- Per-person state: key = (cam_id, track_id) ----
        self.prev_hip_y = {}
        self.prev_time = {}
        self.velocity_y = {}
        self.angle_smooth = defaultdict(list)

        self.baseline_timer = {}
        self.is_baseline_locked = {}
        self.has_been_upright = {}

        self.descent_counter = defaultdict(int)
        self.impact_detected = defaultdict(bool)
        self.fall_timer = {}
        self.inertia_timer = {}
        self.ground_confirm_accum = defaultdict(float)
        self.cooldown_timer = {}

        self.last_seen = {}
        self.last_alert_ts = {}

    def _debug(self, msg):
        if self.debug:
            print(msg)

    @staticmethod
    def _compute_torso_angle(kps):
        shoulder_mid = (kps[5] + kps[6]) / 2
        hip_mid = (kps[11] + kps[12]) / 2
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        return abs(np.degrees(np.arctan2(dx, dy)))

    @staticmethod
    def _compute_body_ratio(kps):
        xs = kps[:, 0]
        ys = kps[:, 1]
        width = np.max(xs) - np.min(xs)
        height = np.max(ys) - np.min(ys)
        if height <= 0:
            return 0.0
        return min(float(width / height), 2.0)

    def _camera_thresholds(self, cam_id):
        """Returns mode-specific thresholds."""
        is_overhead = self.camera_mode.get(cam_id, False)
        if is_overhead:
            return {
                "IS_UPRIGHT_RATIO": 0.60,
                "IMPACT_LATCH_CLEAR_RATIO": 0.25,
                "GROUND_CONFIRM_RATIO": 0.30,
                "MIN_FALL_ANGLE_PATH_B": 60,
                "MIN_TILT_FRAMES_PATH_B": 8,
            }
        return {
            "IS_UPRIGHT_RATIO": 0.52,
            "IMPACT_LATCH_CLEAR_RATIO": 0.30,
            "GROUND_CONFIRM_RATIO": 0.30,
            "MIN_FALL_ANGLE_PATH_B": 25,
            "MIN_TILT_FRAMES_PATH_B": 6,
        }

    def _maybe_finalize_camera_mode(self, cam_id):
        if cam_id in self.camera_mode:
            return

        if self.overhead_camera_override is not None:
            self.camera_mode[cam_id] = bool(self.overhead_camera_override)
            print(f"[FallPlugin] {cam_id} mode forced: {'OVERHEAD' if self.camera_mode[cam_id] else 'SIDE'}")
            return

        ratios = self.mode_ratio_samples[cam_id]
        angles = self.mode_angle_samples[cam_id]
        if len(ratios) < self.AUTO_SAMPLE_DETECTIONS:
            return

        median_ratio = float(np.median(ratios))
        median_angle = float(np.median(angles))
        vote_ratio = median_ratio > 0.55
        vote_angle = median_angle < 45.0
        self.camera_mode[cam_id] = vote_ratio and vote_angle
        print(
            f"[FallPlugin] {cam_id} auto mode: {'OVERHEAD' if self.camera_mode[cam_id] else 'SIDE'} "
            f"(ratio={median_ratio:.3f}, angle={median_angle:.1f})"
        )

    def _velocity_cap(self, person_key):
        if self.descent_counter[person_key] >= self.HOT_DESCENT_FRAMES:
            return self.IMPACT_VEL_CAP_HOT
        return self.IMPACT_VEL_CAP_COLD

    def _cleanup_stale_tracks(self, now_ts, ttl=2.0):
        stale = [k for k, t in self.last_seen.items() if (now_ts - t) > ttl]
        for k in stale:
            self.prev_hip_y.pop(k, None)
            self.prev_time.pop(k, None)
            self.velocity_y.pop(k, None)
            self.angle_smooth.pop(k, None)
            self.baseline_timer.pop(k, None)
            self.is_baseline_locked.pop(k, None)
            self.has_been_upright.pop(k, None)
            self.descent_counter.pop(k, None)
            self.impact_detected.pop(k, None)
            self.fall_timer.pop(k, None)
            self.inertia_timer.pop(k, None)
            self.ground_confirm_accum.pop(k, None)
            self.cooldown_timer.pop(k, None)
            self.last_seen.pop(k, None)

    def process_batch(self, scene_state):
        alerts = []
        now_global = time.time()

        for cam_id, data in scene_state.items():
            tracked = data.get("tracked_people", {})

            for track_id, person in tracked.items():
                person_key = (cam_id, track_id)
                self.last_seen[person_key] = now_global

                bbox = person.get("bbox")
                conf = float(person.get("confidence", 0.0))
                kps = person.get("keypoints")

                if bbox is None or conf < self.conf_threshold:
                    continue

                # Require keypoints for advanced physics-based fall logic.
                if kps is None or not isinstance(kps, np.ndarray) or kps.shape[0] < 17:
                    continue

                xs = kps[:, 0]
                ys = kps[:, 1]
                valid = (xs > 0) & (ys > 0)
                if valid.sum() < 8:
                    continue

                torso_angle_raw = self._compute_torso_angle(kps)
                body_ratio = self._compute_body_ratio(kps)

                smooth_buf = self.angle_smooth[person_key]
                smooth_buf.append(torso_angle_raw)
                if len(smooth_buf) > self.SMOOTH_WINDOW:
                    smooth_buf.pop(0)
                torso_angle = float(np.mean(smooth_buf))

                # Camera mode auto-sampling from valid detections.
                if cam_id not in self.camera_mode:
                    self.mode_ratio_samples[cam_id].append(body_ratio)
                    self.mode_angle_samples[cam_id].append(torso_angle)
                    self._maybe_finalize_camera_mode(cam_id)

                thresholds = self._camera_thresholds(cam_id)
                is_upright = body_ratio < thresholds["IS_UPRIGHT_RATIO"]

                # Hip velocity
                hip_mid_y = float((kps[11][1] + kps[12][1]) / 2.0)
                now_ts = time.time()
                prev_t = self.prev_time.get(person_key, now_ts)
                dt = max(1e-4, now_ts - prev_t)
                self.prev_time[person_key] = now_ts

                prev_hip = self.prev_hip_y.get(person_key, hip_mid_y)
                raw_drop = (hip_mid_y - prev_hip) / dt
                self.prev_hip_y[person_key] = hip_mid_y

                cap = self._velocity_cap(person_key)
                if raw_drop > cap:
                    drop = 0.0
                else:
                    drop = max(0.0, float(raw_drop))
                self.velocity_y[person_key] = drop

                # 1) Baseline lock
                if person_key not in self.baseline_timer:
                    self.baseline_timer[person_key] = now_ts
                    self.is_baseline_locked[person_key] = False
                    self.has_been_upright[person_key] = False

                if not self.is_baseline_locked[person_key]:
                    if now_ts - self.baseline_timer[person_key] > self.BASELINE_TIME:
                        self.is_baseline_locked[person_key] = True
                    continue

                # 2) Upright requirement
                if is_upright:
                    self.has_been_upright[person_key] = True

                if not self.has_been_upright[person_key] and self.descent_counter[person_key] < self.MIN_DESCENT_FRAMES:
                    continue

                # 3) Cooldown
                cool_start = self.cooldown_timer.get(person_key)
                if cool_start is not None and (now_ts - cool_start) < self.RECOVERY_COOLDOWN:
                    continue
                if cool_start is not None and (now_ts - cool_start) >= self.RECOVERY_COOLDOWN:
                    self.cooldown_timer.pop(person_key, None)

                # 4) Descent
                if drop > self.DESCENT_VEL_THRESH:
                    self.descent_counter[person_key] += 1
                else:
                    self.descent_counter[person_key] = 0

                # 5) Impact detection (Path A or Path B)
                if not self.impact_detected[person_key]:
                    path_a = self.IMPACT_VEL_THRESH < drop <= self._velocity_cap(person_key)
                    path_b = (
                        body_ratio > 0.85
                        and torso_angle >= thresholds["MIN_FALL_ANGLE_PATH_B"]
                        and self.descent_counter[person_key] >= self.MIN_DESCENT_FRAMES
                    )

                    if path_a or path_b:
                        self.impact_detected[person_key] = True
                        self.fall_timer[person_key] = now_ts
                        self.inertia_timer[person_key] = now_ts
                        self.ground_confirm_accum[person_key] = 0.0
                        self._debug(
                            f"[FallPlugin] impact cam={cam_id} id={track_id} "
                            f"drop={drop:.1f} ratio={body_ratio:.2f} angle={torso_angle:.1f}"
                        )

                # 6) Ground confirmation
                if self.impact_detected[person_key]:
                    if body_ratio < thresholds["IMPACT_LATCH_CLEAR_RATIO"] and (
                        now_ts - self.inertia_timer.get(person_key, now_ts)
                    ) > self.INERTIA_TIME:
                        self.impact_detected[person_key] = False
                        self.descent_counter[person_key] = 0
                        self.ground_confirm_accum[person_key] = 0.0
                        continue

                    if body_ratio > thresholds["GROUND_CONFIRM_RATIO"]:
                        self.ground_confirm_accum[person_key] += dt

                    if self.ground_confirm_accum[person_key] >= self.GROUND_CONFIRM_TIME:
                        last_alert = self.last_alert_ts.get(person_key, 0.0)
                        if (now_ts - last_alert) >= self.RECOVERY_COOLDOWN:
                            print(
                                f"⚠️  Fall detected on {cam_id} for track {track_id} "
                                f"(conf={conf:.2f}, ratio={body_ratio:.2f})"
                            )
                            alerts.append(
                                {
                                    "cameraId": cam_id,
                                    "eventType": "fall",
                                    "confidence": conf,
                                }
                            )
                            self.last_alert_ts[person_key] = now_ts

                        self.cooldown_timer[person_key] = now_ts
                        self.impact_detected[person_key] = False
                        self.descent_counter[person_key] = 0
                        self.ground_confirm_accum[person_key] = 0.0

                # Recovery reset hint
                if body_ratio < 0.40 and torso_angle < self.UPRIGHT_ANGLE:
                    self.descent_counter[person_key] = 0

        self._cleanup_stale_tracks(now_global)
        return alerts