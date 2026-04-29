import time
from collections import defaultdict, deque

import numpy as np


# ── Fall Severity Levels ───────────────────────────────────────────────────────
class FallSeverity:
    """
    CRITICAL – Person has been on the ground for an extended time (≥3s) with no
               recovery. May be unconscious or unable to get up. Immediate response
               required. Escalated from HIGH if ground dwell time keeps accumulating
               past CRITICAL_GROUND_TIME after the alert fired.
    HIGH     – Fall confirmed: descent + impact + ground confirmation all satisfied.
               Alert is fired, cooldown starts.
    MEDIUM   – Fall in progress: descent frames threshold met, OR impact latched but
               ground confirmation not yet accumulated enough.
    LOW      – No fall detected: below-threshold movement, exercise/seated suppression,
               baseline not locked, or a guard (ghost/edge/area) rejected the detection.
    """
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class RuleBasedFallPlugin:
    # ── Reference resolution all thresholds are calibrated to ─────────────
    REF_WIDTH = 960

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        aspect_ratio_threshold: float = 1.2,
        overhead_camera_override=False,
        frame_width: int = 960,
        frame_height: int = 540,
        debug: bool = False,
    ):
        print("⚡ Initializing Fall Detection Plugin (Advanced Rule-Based v7 — all cameras, low-FPS optimised)...")

        self.conf_threshold           = confidence_threshold
        self.overhead_camera_override = overhead_camera_override
        self.debug                    = debug

        # ── Frame geometry ─────────────────────────────────────────────────
        self.frame_width  = frame_width
        self.frame_height = frame_height
        res_scale         = frame_width / self.REF_WIDTH
        res_scale_sq      = res_scale ** 2

        # ── Timing / motion constants ──────────────────────────────────────
        self.BASELINE_TIME        = 1.5
        self.UPRIGHT_ANGLE        = 18
        self.GROUND_CONFIRM_TIME  = 0.5
        self.CRITICAL_GROUND_TIME = 3.0
        # FIX (low-FPS): reduced from 3 → 1 so a single frame is enough to
        # register descent at 1–2 FPS where 3 consecutive frames = ~2 s.
        self.MIN_DESCENT_FRAMES   = 1
        self.RECOVERY_COOLDOWN    = 5.0
        self.INERTIA_TIME         = 0.6
        self.KP_CONF_THRESH       = 0.30

        # ── FIX A: dt bounds (replaces hardcoded STREAM_FPS) ──────────────
        # dt is clamped to real wall-clock time; no FPS assumption needed.
        self.DT_MIN = 1.0 / 60.0   # 16.7 ms  — prevents division by near-zero
        self.DT_MAX = 1.0 / 8.0    # 125  ms  — caps outlier gaps (re-ID, pause)

        # ── Velocity thresholds ────────────────────────────────────────────
        self.DESCENT_VEL_THRESH   = 90.0    * res_scale
        self.IMPACT_VEL_THRESH    = 180.0   * res_scale
        self.IMPACT_VEL_CAP_COLD  = 12000.0 * res_scale
        self.IMPACT_VEL_CAP_HOT   = 20000.0 * res_scale

        # ── Exercise / seated suppression ─────────────────────────────────
        self.EXERCISE_MAX_PEAK_DROP     = 900.0 * res_scale
        self.EXERCISE_SLOW_DESCENT_TIME = 0.40
        self.PEAK_DROP_WINDOW           = 40

        # ── Ghost / partial-detection guards ──────────────────────────────
        self.MAX_HIP_JUMP_PX = 500.0  * res_scale
        self.EDGE_MARGIN     = max(15, int(40 * res_scale))
        self.MIN_PERSON_AREA = max(500, int(4000 * res_scale_sq))

        # ── Smoothing ─────────────────────────────────────────────────────
        # FIX (low-FPS): reduced windows from 5→2 and 4→2.
        # At 1.7 FPS, window=5 spans ~3 s which is too much lag.
        self.SMOOTH_WINDOW     = 2   # was 5
        self.HIP_SMOOTH_WINDOW = 2   # was 4
        self.CRITICAL_KPS      = [5, 6, 11, 12, 13, 14, 15, 16]

        # ── FIX E: hip-rising guard ────────────────────────────────────────
        self.HIP_RISING_GUARD_PX = 15
        self.hip_y_smooth        = defaultdict(list)

        # ── Camera auto-detection ─────────────────────────────────────────
        self.AUTO_SAMPLE_DETECTIONS = 80
        self.mode_ratio_samples = defaultdict(list)
        self.mode_angle_samples = defaultdict(list)
        self.camera_mode        = {}

        # ── Per-person state ───────────────────────────────────────────────
        self.prev_hip_y          = {}
        self.prev_time           = {}
        self.velocity_y          = {}
        self.angle_smooth        = defaultdict(list)

        self.baseline_timer      = {}
        self.is_baseline_locked  = {}
        self.has_been_upright    = {}

        self.descent_counter      = defaultdict(int)
        self.impact_detected      = defaultdict(bool)
        self.fall_timer           = {}
        self.inertia_timer        = {}
        self.ground_confirm_accum = defaultdict(float)
        self.cooldown_timer       = {}

        self.tilt_start_time  = {}
        self.tilt_frame_count = defaultdict(int)
        self.all_drop_history = {}

        self.falling_ids = set()

        self.last_seen     = {}
        self.last_alert_ts = {}
        self.alert_time    = {}

        self.current_severity = {}

        # ── v7: overhead + frontal fall history buffers ───────────────────
        # FIX (low-FPS): maxlen reduced from 6 → 3.
        # At 1.7 FPS, maxlen=6 spans 3.5 s — way too long a history window.
        # Path C — ceiling camera: tracks bbox area expansion + keypoint spread
        # Path D — frontal fall:   tracks bbox diagonal growth + centre drop
        self._overhead_history = {}   # person_key -> deque[(ts, area, spread)]
        self._frontal_history  = {}   # person_key -> deque[(ts, diag, cy)]

        # ── FPS tracking ──────────────────────────────────────────────────
        self.fps_times = deque(maxlen=30)

        self._debug(
            f"[FallPlugin] frame={frame_width}×{frame_height}  res_scale={res_scale:.3f}\n"
            f"  DESCENT_VEL_THRESH   = {self.DESCENT_VEL_THRESH:.0f} px/s\n"
            f"  IMPACT_VEL_THRESH    = {self.IMPACT_VEL_THRESH:.0f} px/s\n"
            f"  IMPACT_VEL_CAP_COLD  = {self.IMPACT_VEL_CAP_COLD:.0f} px/s\n"
            f"  IMPACT_VEL_CAP_HOT   = {self.IMPACT_VEL_CAP_HOT:.0f} px/s\n"
            f"  MAX_HIP_JUMP_PX      = {self.MAX_HIP_JUMP_PX:.0f} px\n"
            f"  EDGE_MARGIN          = {self.EDGE_MARGIN} px\n"
            f"  MIN_PERSON_AREA      = {self.MIN_PERSON_AREA} px²\n"
            f"  GROUND_CONFIRM_TIME  = {self.GROUND_CONFIRM_TIME:.2f} s\n"
            f"  MIN_DESCENT_FRAMES   = {self.MIN_DESCENT_FRAMES}  ← low-FPS fix\n"
            f"  SMOOTH_WINDOW        = {self.SMOOTH_WINDOW}  ← low-FPS fix\n"
            f"  HIP_SMOOTH_WINDOW    = {self.HIP_SMOOTH_WINDOW}  ← low-FPS fix\n"
            f"  HIP_RISING_GUARD_PX  = {self.HIP_RISING_GUARD_PX} px\n"
            f"  DT_MIN               = {self.DT_MIN*1000:.1f} ms  ← replaces STREAM_FPS\n"
            f"  DT_MAX               = {self.DT_MAX*1000:.0f} ms\n"
            f"  overhead_override    = {self.overhead_camera_override}\n"
            f"  [v7] Path C (ceiling) + Path D (frontal) active\n"
            f"  [v7] history buffers maxlen=3  ← low-FPS fix"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _debug(self, msg: str):
        if self.debug:
            print(msg, flush=True)

    @staticmethod
    def _compute_torso_angle(kps: np.ndarray) -> float:
        shoulder_mid = (kps[5] + kps[6]) / 2
        hip_mid      = (kps[11] + kps[12]) / 2
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        return float(abs(np.degrees(np.arctan2(dx, dy))))

    @staticmethod
    def _compute_body_ratio(kps: np.ndarray) -> float:
        xs     = kps[:, 0]
        ys     = kps[:, 1]
        width  = np.max(xs) - np.min(xs)
        height = np.max(ys) - np.min(ys)
        if height <= 0:
            return 0.0
        return float(min(width / height, 2.0))

    def _camera_thresholds(self, cam_id: str) -> dict:
        is_overhead = self.camera_mode.get(cam_id, False)
        if is_overhead:
            return {
                "IS_UPRIGHT_RATIO":          0.60,
                "IMPACT_LATCH_CLEAR_RATIO":  0.25,
                "GROUND_CONFIRM_RATIO":      0.45,
                "MIN_FALL_ANGLE_PATH_B":     60,
                "MIN_TILT_FRAMES_PATH_B":    15,
                "HOT_DESCENT_FRAMES":        5,
                "EXERCISE_MIN_RATIO_FRAMES": 10,
            }
        return {
            "IS_UPRIGHT_RATIO":          0.52,
            "IMPACT_LATCH_CLEAR_RATIO":  0.30,
            "GROUND_CONFIRM_RATIO":      0.50,
            "PATH_B_RATIO_THRESH":       0.80,
            "MIN_FALL_ANGLE_PATH_B":     25,
            "MIN_TILT_FRAMES_PATH_B":    12,
            "HOT_DESCENT_FRAMES":        3,
            "EXERCISE_MIN_RATIO_FRAMES": 15,
        }

    def _maybe_finalize_camera_mode(self, cam_id: str):
        if cam_id in self.camera_mode:
            return

        if self.overhead_camera_override is not None:
            self.camera_mode[cam_id] = bool(self.overhead_camera_override)
            self._debug(
                f"[FallPlugin] {cam_id} mode FORCED: "
                f"{'OVERHEAD' if self.camera_mode[cam_id] else 'SIDE'}"
            )
            return

        ratios = self.mode_ratio_samples[cam_id]
        angles = self.mode_angle_samples[cam_id]
        if len(ratios) < self.AUTO_SAMPLE_DETECTIONS:
            return

        median_ratio = float(np.median(ratios))
        median_angle = float(np.median(angles))
        vote_ratio   = median_ratio > 0.55
        vote_angle   = median_angle < 45.0
        self.camera_mode[cam_id] = vote_ratio and vote_angle
        self._debug(
            f"[FallPlugin] {cam_id} AUTO mode: "
            f"{'OVERHEAD' if self.camera_mode[cam_id] else 'SIDE'} "
            f"(ratio={median_ratio:.3f}, angle={median_angle:.1f}°)"
        )

    def _velocity_cap(self, person_key, hot_descent_frames: int) -> float:
        if self.descent_counter[person_key] >= hot_descent_frames:
            return self.IMPACT_VEL_CAP_HOT
        return self.IMPACT_VEL_CAP_COLD

    def _is_exercise_motion(self, person_key, exercise_min_ratio_frames: int) -> bool:
        history     = self.all_drop_history.get(person_key)
        t_start     = self.tilt_start_time.get(person_key)
        frame_count = self.tilt_frame_count[person_key]

        if not history or t_start is None:
            return False

        max_drop = max(history) if history else 0
        elapsed  = time.time() - t_start

        no_spike      = max_drop    < self.EXERCISE_MAX_PEAK_DROP
        gradual_ratio = frame_count > exercise_min_ratio_frames

        if frame_count > 25 and elapsed > 1.0 and no_spike:
            self._debug(
                f"{person_key} 🪑 SEATED SUPPRESSION "
                f"(frames={frame_count} elapsed={elapsed:.2f}s max_drop={max_drop:.0f})"
            )
            return True

        if not no_spike:
            self._debug(
                f"{person_key} ⚡ hard spike (max_drop={max_drop:.0f}) — not exercise"
            )
            return False

        if gradual_ratio:
            self._debug(
                f"{person_key} 🏋️  EXERCISE SUPPRESSION "
                f"(no_spike=True gradual={gradual_ratio} "
                f"max_drop={max_drop:.0f} elapsed={elapsed:.2f}s frames={frame_count})"
            )
            return True
        return False

    # ── v7: Path C — ceiling/overhead camera ──────────────────────────────
    def _compute_overhead_signals(self, person_key, kps_xy, bbox, dt, now_ts):
        """
        For ceiling cameras hip Y velocity is useless — the person falls
        *away* from the lens.  Instead we watch:

        1. BBOX AREA EXPANSION RATE
           Standing from above → small bbox.  Lying on floor → large wide bbox.
           Rapid expansion = fall impact.

        2. KEYPOINT SPREAD (std-dev of all visible keypoints)
           Upright: keypoints clustered.  On impact: limbs splay outward.

        Returns {"area_rate", "spread_rate", "fired"}
        """
        x1, y1, x2, y2 = bbox
        area_now = float((x2 - x1) * (y2 - y1))

        xs, ys = kps_xy[:, 0], kps_xy[:, 1]
        valid  = (xs > 0) & (ys > 0)
        spread = float(np.std(kps_xy[valid])) if valid.sum() > 3 else 0.0

        # FIX (low-FPS): maxlen reduced 6 → 3
        history = self._overhead_history.setdefault(person_key, deque(maxlen=3))
        history.append((now_ts, area_now, spread))

        if len(history) < 3:
            return {"area_rate": 0.0, "spread_rate": 0.0, "fired": False}

        t0, area0, spread0 = history[0]
        elapsed = max(now_ts - t0, dt)

        area_rate   = (area_now - area0)  / elapsed
        spread_rate = (spread   - spread0) / elapsed

        AREA_RATE_THRESH   = 30_000.0 * (self.frame_width / self.REF_WIDTH)
        SPREAD_RATE_THRESH =     60.0 * (self.frame_width / self.REF_WIDTH)

        fired = (area_rate > AREA_RATE_THRESH) or (spread_rate > SPREAD_RATE_THRESH)

        if self.debug and fired:
            print(
                f"[OverheadFix] Path-C FIRED person={person_key} "
                f"area_rate={area_rate:.0f} spread_rate={spread_rate:.1f}",
                flush=True,
            )

        return {"area_rate": area_rate, "spread_rate": spread_rate, "fired": fired}

    # ── v7: Path D — frontal fall (person falls toward camera) ─────────────
    def _compute_frontal_signals(self, person_key, bbox, dt, now_ts):
        """
        When someone falls toward the camera their bbox diagonal grows rapidly
        AND the bbox centre drops (head pitching forward/down).

        Both conditions must fire together to avoid false positives from
        someone simply walking toward the camera.

        Returns {"diag_rate", "centre_drop", "fired"}
        """
        x1, y1, x2, y2 = bbox
        diag = float(np.hypot(x2 - x1, y2 - y1))
        cy   = float((y1 + y2) / 2.0)

        # FIX (low-FPS): maxlen reduced 6 → 3
        history = self._frontal_history.setdefault(person_key, deque(maxlen=3))
        history.append((now_ts, diag, cy))

        if len(history) < 3:
            return {"diag_rate": 0.0, "centre_drop": 0.0, "fired": False}

        t0, diag0, cy0 = history[0]
        elapsed = max(now_ts - t0, dt)

        diag_rate   = (diag - diag0) / elapsed
        centre_drop = (cy   - cy0)   / elapsed   # positive = moving down

        DIAG_RATE_THRESH   = 400.0 * (self.frame_width / self.REF_WIDTH)
        CENTRE_DROP_THRESH =  60.0 * (self.frame_width / self.REF_WIDTH)

        fired = (diag_rate > DIAG_RATE_THRESH) and (centre_drop > CENTRE_DROP_THRESH)

        if self.debug and fired:
            print(
                f"[FrontalFix] Path-D FIRED person={person_key} "
                f"diag_rate={diag_rate:.0f} centre_drop={centre_drop:.1f}",
                flush=True,
            )

        return {"diag_rate": diag_rate, "centre_drop": centre_drop, "fired": fired}

    def _cleanup_stale_tracks(self, now_ts: float, ttl: float = 2.0):
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
            self.tilt_start_time.pop(k, None)
            self.tilt_frame_count.pop(k, None)
            self.all_drop_history.pop(k, None)
            self.falling_ids.discard(k)
            self.current_severity.pop(k, None)
            self.alert_time.pop(k, None)
            self.hip_y_smooth.pop(k, None)
            self._overhead_history.pop(k, None)
            self._frontal_history.pop(k, None)
            self.last_seen.pop(k, None)

    def _compute_severity(self, person_key) -> str:
        descent_ok = self.descent_counter[person_key] >= self.MIN_DESCENT_FRAMES
        impact_ok  = self.impact_detected[person_key]
        ground_ok  = self.ground_confirm_accum[person_key] >= self.GROUND_CONFIRM_TIME

        alert_ts = self.alert_time.get(person_key)
        if alert_ts is not None:
            if (time.time() - alert_ts) >= self.CRITICAL_GROUND_TIME:
                return FallSeverity.CRITICAL

        if impact_ok and ground_ok:
            return FallSeverity.HIGH
        if descent_ok or impact_ok:
            return FallSeverity.MEDIUM
        return FallSeverity.LOW

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def get_severity(self, cam_id: str, track_id) -> str:
        return self.current_severity.get((cam_id, track_id), FallSeverity.LOW)

    def process_batch(self, scene_state: dict) -> list:
        """
        scene_state: {
            cam_id: {
                "tracked_people": {
                    track_id: {
                        "bbox":       [x1,y1,x2,y2],
                        "confidence": float,
                        "keypoints":  array[17,2|3],
                        # optional:
                        "frame_width":  int,
                        "frame_height": int,
                    }
                }
            }
        }

        Returns list of alert dicts:
            {
                "cameraId":   str,
                "trackId":    any,
                "eventType":  "fall",
                "confidence": float,
                "severity":   "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
            }

        Fall detection paths (v7):
            Path A — velocity spike (side/front cameras, fast fall)
            Path B — body goes horizontal + tilt angle (front camera, sideways fall)
            Path C — bbox area + keypoint spread (ceiling/overhead camera)
            Path D — bbox diagonal growth + centre drop (frontal fall)
        """
        # ── FPS tracking ──────────────────────────────────────────────────
        self.fps_times.append(time.time())
        if len(self.fps_times) == 30:
            fps = 30 / (self.fps_times[-1] - self.fps_times[0])
            print(f"[FPS] {fps:.1f}", flush=True)
        # ──────────────────────────────────────────────────────────────────

        alerts     = []
        now_global = time.time()

        for cam_id, data in scene_state.items():
            tracked = data.get("tracked_people", {})

            for track_id, person in tracked.items():
                person_key = (cam_id, track_id)
                self.last_seen[person_key] = now_global

                bbox    = person.get("bbox")
                conf    = float(person.get("confidence", 0.0))
                kps_raw = person.get("keypoints")

                fw = int(person.get("frame_width",  self.frame_width))
                fh = int(person.get("frame_height", self.frame_height))

                self._debug(
                    f"[FallPlugin] INTAKE cam={cam_id} id={track_id} "
                    f"conf={conf:.2f} bbox={bbox is not None} "
                    f"kps_type={type(kps_raw).__name__} "
                    f"kps_shape={np.array(kps_raw).shape if kps_raw is not None else 'None'}"
                )

                # ── Basic validity checks ──────────────────────────────────
                if bbox is None:
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                if conf < self.conf_threshold:
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                if kps_raw is None:
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                kps = np.array(kps_raw, dtype=float)

                if kps.ndim != 2 or kps.shape[0] < 17:
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                kps_xy = kps[:, :2]

                # ── Keypoint confidence check ──────────────────────────────
                if kps.shape[1] >= 3:
                    kps_conf      = kps[:, 2]
                    critical_conf = float(np.mean(kps_conf[self.CRITICAL_KPS]))
                    if critical_conf < self.KP_CONF_THRESH:
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue
                else:
                    xs    = kps_xy[:, 0]
                    ys    = kps_xy[:, 1]
                    valid = (xs > 0) & (ys > 0)
                    if int(valid.sum()) < 8:
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue

                # ── Ghost / edge / area guards ─────────────────────────────
                x1b, y1b, x2b, y2b = bbox
                cx = (x1b + x2b) / 2
                cy = (y1b + y2b) / 2
                em = self.EDGE_MARGIN

                if cx < em or cx > fw - em or cy < em or cy > fh - em:
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                bbox_area = (x2b - x1b) * (y2b - y1b)
                if bbox_area < self.MIN_PERSON_AREA:
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                # ── Torso geometry ─────────────────────────────────────────
                torso_angle_raw = self._compute_torso_angle(kps_xy)
                body_ratio      = self._compute_body_ratio(kps_xy)

                smooth_buf = self.angle_smooth[person_key]
                smooth_buf.append(torso_angle_raw)
                if len(smooth_buf) > self.SMOOTH_WINDOW:
                    smooth_buf.pop(0)
                torso_angle = float(np.mean(smooth_buf))

                # ── Camera mode sampling / finalization ────────────────────
                if cam_id not in self.camera_mode:
                    self.mode_ratio_samples[cam_id].append(body_ratio)
                    self.mode_angle_samples[cam_id].append(torso_angle)
                    self._maybe_finalize_camera_mode(cam_id)

                if cam_id not in self.camera_mode:
                    self._debug(
                        f"[FallPlugin] cam={cam_id} mode not yet locked "
                        f"({len(self.mode_ratio_samples[cam_id])}/{self.AUTO_SAMPLE_DETECTIONS} samples)"
                        f" — using side-view defaults"
                    )

                thresholds                = self._camera_thresholds(cam_id)
                hot_descent_frames        = thresholds["HOT_DESCENT_FRAMES"]
                exercise_min_ratio_frames = thresholds["EXERCISE_MIN_RATIO_FRAMES"]
                is_overhead               = self.camera_mode.get(cam_id, False)
                is_upright                = body_ratio < thresholds["IS_UPRIGHT_RATIO"]

                path_b_ratio_thresh = thresholds.get(
                    "PATH_B_RATIO_THRESH",
                    0.90 if is_overhead else 0.80
                )

                # ── Hip velocity ───────────────────────────────────────────
                hip_mid_y_raw = float((kps_xy[11][1] + kps_xy[12][1]) / 2.0)

                hip_buf = self.hip_y_smooth[person_key]
                hip_buf.append(hip_mid_y_raw)
                if len(hip_buf) > self.HIP_SMOOTH_WINDOW:
                    hip_buf.pop(0)
                hip_mid_y = float(np.mean(hip_buf))

                now_ts = time.time()
                prev_t = self.prev_time.get(person_key, now_ts)

                # ── FIX A: real wall-clock dt, no STREAM_FPS assumption ────
                raw_dt = now_ts - prev_t
                dt     = float(np.clip(raw_dt, self.DT_MIN, self.DT_MAX))

                self.prev_time[person_key] = now_ts

                if person_key not in self.prev_hip_y:
                    self.prev_hip_y[person_key] = hip_mid_y
                    self.velocity_y[person_key] = 0.0
                    prev_hip = hip_mid_y
                else:
                    prev_hip = self.prev_hip_y[person_key]

                # ── Hip-jump guard ─────────────────────────────────────────
                hip_jump = abs(hip_mid_y - prev_hip)
                if hip_jump > self.MAX_HIP_JUMP_PX:
                    if not self.is_baseline_locked.get(person_key, False):
                        self.prev_hip_y[person_key] = hip_mid_y
                        self.velocity_y[person_key] = 0.0
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue

                raw_drop = (hip_mid_y - prev_hip) / dt
                self.prev_hip_y[person_key] = hip_mid_y

                cap = self._velocity_cap(person_key, hot_descent_frames)
                if raw_drop > cap:
                    drop = 0.0
                elif raw_drop < 0:
                    drop = 0.0
                else:
                    drop = float(raw_drop)

                self.velocity_y[person_key] = drop

                self._debug(
                    f"[FallPlugin] PHYSICS cam={cam_id} id={track_id} "
                    f"ratio={body_ratio:.2f} angle={torso_angle:.1f} "
                    f"drop={drop:.1f} dt={dt*1000:.1f}ms is_upright={is_upright} "
                    f"descent={self.descent_counter[person_key]} "
                    f"impact={self.impact_detected[person_key]} "
                    f"baseline_locked={self.is_baseline_locked.get(person_key, False)}"
                )

                # ══════════════════════════════════════════════════════════
                # 1️⃣  BASELINE LOCK
                # ══════════════════════════════════════════════════════════
                if person_key not in self.baseline_timer:
                    self.baseline_timer[person_key]     = now_ts
                    self.is_baseline_locked[person_key] = False
                    self.has_been_upright[person_key]   = False

                if not self.is_baseline_locked[person_key]:
                    if now_ts - self.baseline_timer[person_key] > self.BASELINE_TIME:
                        self.is_baseline_locked[person_key] = True
                        self._debug(f"[FallPlugin] baseline locked cam={cam_id} id={track_id}")
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                # ══════════════════════════════════════════════════════════
                # 2️⃣  UPRIGHT TRACKING + TILT CLOCK
                # ══════════════════════════════════════════════════════════
                if is_upright:
                    self.has_been_upright[person_key] = True
                    self.tilt_start_time.pop(person_key, None)
                    self.all_drop_history.pop(person_key, None)
                    self.tilt_frame_count[person_key] = 0

                    if self.impact_detected[person_key]:
                        inertia_elapsed = now_ts - self.inertia_timer.get(person_key, now_ts)
                        if inertia_elapsed > self.INERTIA_TIME:
                            if body_ratio < thresholds["IMPACT_LATCH_CLEAR_RATIO"]:
                                self.impact_detected[person_key]      = False
                                self.descent_counter[person_key]      = 0
                                self.ground_confirm_accum[person_key] = 0.0
                                self.alert_time.pop(person_key, None)
                                self._debug(
                                    f"[FallPlugin] impact latch cleared (upright) "
                                    f"cam={cam_id} id={track_id}"
                                )
                else:
                    if person_key not in self.tilt_start_time and not self.impact_detected[person_key]:
                        self.tilt_start_time[person_key]  = now_ts
                        self.all_drop_history[person_key] = deque(maxlen=self.PEAK_DROP_WINDOW)
                        self.tilt_frame_count[person_key] = 0

                    if person_key in self.tilt_start_time:
                        self.tilt_frame_count[person_key] = min(
                            self.tilt_frame_count[person_key] + 1, 60
                        )
                        if drop > 0:
                            self.all_drop_history[person_key].append(drop)

                # ══════════════════════════════════════════════════════════
                # 3️⃣  RECOVERY COOLDOWN
                # ══════════════════════════════════════════════════════════
                cool_start = self.cooldown_timer.get(person_key)
                if cool_start is not None:
                    if (now_ts - cool_start) < self.RECOVERY_COOLDOWN:
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue
                    else:
                        self.cooldown_timer.pop(person_key, None)
                        self.impact_detected[person_key] = False
                        self.descent_counter[person_key] = 0

                # ══════════════════════════════════════════════════════════
                # 4️⃣  DESCENT DETECTION
                # ══════════════════════════════════════════════════════════
                if drop > self.DESCENT_VEL_THRESH:
                    self.descent_counter[person_key] += 1
                else:
                    self.descent_counter[person_key] = 0

                not_enough_descent = self.descent_counter[person_key] < self.MIN_DESCENT_FRAMES
                if not_enough_descent:
                    self.falling_ids.discard(person_key)
                    if not self.impact_detected[person_key]:
                        self.current_severity[person_key] = FallSeverity.LOW
                        # Fall through to Path C / D checks below instead of hard continue.
                        # We use a flag to suppress Path A / B if descent not met.
                        skip_velocity_paths = True
                    else:
                        skip_velocity_paths = False
                else:
                    self.falling_ids.add(person_key)
                    skip_velocity_paths = False

                # ══════════════════════════════════════════════════════════
                # 5️⃣  IMPACT DETECTION
                #     Path A — velocity spike
                #     Path B — body goes horizontal (front cam, sideways fall)
                #     Path C — ceiling/overhead expansion
                #     Path D — frontal fall (bbox grows + centre drops)
                # ══════════════════════════════════════════════════════════
                if not self.impact_detected[person_key]:
                    ratio_now   = body_ratio
                    tilt_frames = self.tilt_frame_count[person_key]
                    current_cap = self._velocity_cap(person_key, hot_descent_frames)

                    # ── Path A — velocity spike ────────────────────────────
                    if not skip_velocity_paths and self.IMPACT_VEL_THRESH < drop <= current_cap:
                        path_a_blocked = False

                        if is_overhead:
                            tilt_elapsed_now = now_ts - self.tilt_start_time.get(person_key, now_ts)
                            seated_at_spike  = (ratio_now >= 0.60 and tilt_elapsed_now < 0.5)
                            long_seated_ctx  = self._is_exercise_motion(
                                person_key, exercise_min_ratio_frames
                            )
                            if seated_at_spike or long_seated_ctx:
                                path_a_blocked = True

                        if not path_a_blocked:
                            self._latch_impact(person_key, now_ts)
                            self._debug(
                                f"[FallPlugin] 💥 IMPACT-VELOCITY (Path A) "
                                f"cam={cam_id} id={track_id} drop={drop:.0f}"
                            )

                    # ── Path B — body horizontal + tilt (front cam, sideways fall)
                    elif (
                        not skip_velocity_paths
                        and ratio_now > path_b_ratio_thresh
                        and torso_angle >= thresholds["MIN_FALL_ANGLE_PATH_B"]
                        and tilt_frames >= thresholds["MIN_TILT_FRAMES_PATH_B"]
                        and self.descent_counter[person_key] >= self.MIN_DESCENT_FRAMES
                    ):
                        if self._is_exercise_motion(person_key, exercise_min_ratio_frames):
                            self.falling_ids.discard(person_key)
                            self.current_severity[person_key] = FallSeverity.LOW
                            continue
                        else:
                            self._latch_impact(person_key, now_ts)
                            self._debug(
                                f"[FallPlugin] 💥 IMPACT-RATIO (Path B) "
                                f"cam={cam_id} id={track_id} "
                                f"ratio={ratio_now:.2f} angle={torso_angle:.1f} "
                                f"frames={tilt_frames}"
                            )

                    # ── Path C — ceiling/overhead: area expansion + kp spread ──
                    if is_overhead and not self.impact_detected[person_key]:
                        oh = self._compute_overhead_signals(
                            person_key, kps_xy, bbox, dt, now_ts
                        )
                        if oh["fired"]:
                            self._latch_impact(person_key, now_ts)
                            self._debug(
                                f"[FallPlugin] 💥 IMPACT-OVERHEAD (Path C) "
                                f"cam={cam_id} id={track_id} "
                                f"area_rate={oh['area_rate']:.0f} "
                                f"spread_rate={oh['spread_rate']:.1f}"
                            )

                    # ── Path D — frontal: bbox diagonal growth + centre drop ────
                    if not is_overhead and not self.impact_detected[person_key]:
                        fr = self._compute_frontal_signals(
                            person_key, bbox, dt, now_ts
                        )
                        if fr["fired"]:
                            self._latch_impact(person_key, now_ts)
                            self._debug(
                                f"[FallPlugin] 💥 IMPACT-FRONTAL (Path D) "
                                f"cam={cam_id} id={track_id} "
                                f"diag_rate={fr['diag_rate']:.0f} "
                                f"centre_drop={fr['centre_drop']:.1f}"
                            )

                # ── If still no impact after all paths, bail out ───────────
                if not self.impact_detected[person_key]:
                    if skip_velocity_paths:
                        self.current_severity[person_key] = FallSeverity.LOW
                    continue

                # ══════════════════════════════════════════════════════════
                # 6️⃣  GROUND CONFIRMATION
                # ══════════════════════════════════════════════════════════
                inertia_elapsed = now_ts - self.inertia_timer.get(person_key, now_ts)
                if (
                    body_ratio < thresholds["IMPACT_LATCH_CLEAR_RATIO"]
                    and inertia_elapsed > self.INERTIA_TIME
                ):
                    self.impact_detected[person_key]      = False
                    self.descent_counter[person_key]      = 0
                    self.ground_confirm_accum[person_key] = 0.0
                    self.alert_time.pop(person_key, None)
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                if body_ratio > thresholds["GROUND_CONFIRM_RATIO"]:
                    hip_rising = (hip_mid_y < prev_hip - self.HIP_RISING_GUARD_PX)
                    if hip_rising:
                        self.ground_confirm_accum[person_key] = 0.0
                        self._debug(
                            f"[FallPlugin] hip rising (>{self.HIP_RISING_GUARD_PX}px), "
                            f"ground accum reset cam={cam_id} id={track_id}"
                        )
                    else:
                        self.ground_confirm_accum[person_key] += dt

                self._debug(
                    f"[FallPlugin] GROUND_DEBUG cam={cam_id} id={track_id} "
                    f"accum={self.ground_confirm_accum[person_key]:.3f}s "
                    f"ratio={body_ratio:.2f} "
                    f"(confirm when accum>{self.GROUND_CONFIRM_TIME} "
                    f"& ratio>{thresholds['GROUND_CONFIRM_RATIO']})"
                )

                self.current_severity[person_key] = self._compute_severity(person_key)

                if self.ground_confirm_accum[person_key] >= self.GROUND_CONFIRM_TIME:
                    last_alert = self.last_alert_ts.get(person_key, 0.0)
                    if (now_ts - last_alert) >= self.RECOVERY_COOLDOWN:
                        severity = FallSeverity.HIGH
                        self.current_severity[person_key] = severity
                        self.alert_time[person_key]       = now_ts
                        print(
                            f"⚠️  Fall confirmed cam={cam_id} track={track_id} "
                            f"severity={severity} "
                            f"(conf={conf:.2f} ratio={body_ratio:.2f})",
                            flush=True,
                        )
                        alerts.append({
                            "cameraId":   cam_id,
                            "trackId":    track_id,
                            "eventType":  "fall",
                            "confidence": conf,
                            "severity":   severity,
                        })
                        self.last_alert_ts[person_key] = now_ts

                    self.cooldown_timer[person_key]       = now_ts
                    self.impact_detected[person_key]      = False
                    self.descent_counter[person_key]      = 0
                    self.ground_confirm_accum[person_key] = 0.0
                    self.falling_ids.discard(person_key)
                    continue

                # ── CRITICAL escalation ────────────────────────────────────
                alert_ts = self.alert_time.get(person_key)
                if alert_ts is not None and not self.impact_detected[person_key]:
                    time_on_ground = now_ts - alert_ts
                    if time_on_ground >= self.CRITICAL_GROUND_TIME:
                        last_crit = self.last_alert_ts.get(person_key, 0.0)
                        if (now_ts - last_crit) >= self.RECOVERY_COOLDOWN:
                            self.current_severity[person_key] = FallSeverity.CRITICAL
                            print(
                                f"🚨 CRITICAL fall cam={cam_id} track={track_id} "
                                f"— on ground {time_on_ground:.1f}s "
                                f"(conf={conf:.2f} ratio={body_ratio:.2f})",
                                flush=True,
                            )
                            alerts.append({
                                "cameraId":   cam_id,
                                "trackId":    track_id,
                                "eventType":  "fall",
                                "confidence": conf,
                                "severity":   FallSeverity.CRITICAL,
                            })
                            self.last_alert_ts[person_key] = now_ts

                self.current_severity[person_key] = self._compute_severity(person_key)

                if body_ratio < 0.40 and torso_angle < self.UPRIGHT_ANGLE:
                    self.descent_counter[person_key] = 0
                    self.current_severity[person_key] = FallSeverity.LOW

        self._cleanup_stale_tracks(now_global)
        return alerts

    # ──────────────────────────────────────────────────────────────────────
    # Helper: latch impact state for a person
    # ──────────────────────────────────────────────────────────────────────

    def _latch_impact(self, person_key, now_ts: float):
        self.impact_detected[person_key]      = True
        self.fall_timer[person_key]           = now_ts
        self.inertia_timer[person_key]        = now_ts
        self.ground_confirm_accum[person_key] = 0.0
        self.tilt_start_time.pop(person_key, None)
        self.all_drop_history.pop(person_key, None)
        self.tilt_frame_count[person_key]     = 0
        self.current_severity[person_key]     = FallSeverity.MEDIUM