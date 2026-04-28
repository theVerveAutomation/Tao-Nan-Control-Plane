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
        print("⚡ Initializing Fall Detection Plugin (Advanced Rule-Based v5)...")

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
        self.GROUND_CONFIRM_TIME  = 0.5    # raised from 0.06 — requires ~12 frames of lying flat
        self.CRITICAL_GROUND_TIME = 3.0    # seconds on ground after HIGH alert → escalate to CRITICAL
        self.MIN_DESCENT_FRAMES   = 3      # raised from 2 — filters single-frame velocity spikes
        self.RECOVERY_COOLDOWN    = 5.0
        self.INERTIA_TIME         = 0.6
        self.KP_CONF_THRESH       = 0.30
        self.STREAM_FPS           = 25.0

        # ── Velocity thresholds (scaled to actual resolution) ──────────────
        self.DESCENT_VEL_THRESH   = 90.0    * res_scale
        self.IMPACT_VEL_THRESH    = 180.0   * res_scale
        self.IMPACT_VEL_CAP_COLD  = 5000.0  * res_scale
        self.IMPACT_VEL_CAP_HOT   = 20000.0 * res_scale

        # ── Exercise / seated suppression ─────────────────────────────────
        self.EXERCISE_MAX_PEAK_DROP     = 900.0 * res_scale  # tightened from 1200 — catches slower seated drops
        self.EXERCISE_SLOW_DESCENT_TIME = 0.40               # tightened from 0.55 — suppress exercise sooner
        self.PEAK_DROP_WINDOW           = 40

        # ── Ghost / partial-detection guards (scaled) ──────────────────────
        self.MAX_HIP_JUMP_PX = 500.0  * res_scale
        self.EDGE_MARGIN     = max(15, int(40 * res_scale))
        self.MIN_PERSON_AREA = max(500, int(4000 * res_scale_sq))

        # ── Smoothing ─────────────────────────────────────────────────────
        self.SMOOTH_WINDOW = 5
        self.CRITICAL_KPS  = [5, 6, 11, 12, 13, 14, 15, 16]

        # ── Camera auto-detection ─────────────────────────────────────────
        self.AUTO_SAMPLE_DETECTIONS = 80
        self.mode_ratio_samples = defaultdict(list)
        self.mode_angle_samples = defaultdict(list)
        self.camera_mode        = {}

        # ── Per-person state (keyed by (cam_id, track_id)) ─────────────────
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
        self.alert_time    = {}  # person_key -> timestamp when HIGH alert fired (for CRITICAL escalation)

        # ── Per-person severity tracking ───────────────────────────────────
        self.current_severity = {}   # person_key -> FallSeverity.*

        self._debug(
            f"[FallPlugin] frame={frame_width}×{frame_height}  res_scale={res_scale:.3f}\n"
            f"  DESCENT_VEL_THRESH  = {self.DESCENT_VEL_THRESH:.0f} px/s\n"
            f"  IMPACT_VEL_THRESH   = {self.IMPACT_VEL_THRESH:.0f} px/s\n"
            f"  IMPACT_VEL_CAP_COLD = {self.IMPACT_VEL_CAP_COLD:.0f} px/s\n"
            f"  IMPACT_VEL_CAP_HOT  = {self.IMPACT_VEL_CAP_HOT:.0f} px/s\n"
            f"  MAX_HIP_JUMP_PX     = {self.MAX_HIP_JUMP_PX:.0f} px\n"
            f"  EDGE_MARGIN         = {self.EDGE_MARGIN} px\n"
            f"  MIN_PERSON_AREA     = {self.MIN_PERSON_AREA} px²\n"
            f"  GROUND_CONFIRM_TIME = {self.GROUND_CONFIRM_TIME:.2f} s  [raised from 0.06]\n"
            f"  MIN_DESCENT_FRAMES  = {self.MIN_DESCENT_FRAMES}          [raised from 2]\n"
            f"  PATH_B_RATIO_THRESH = 0.90                [raised from 0.85]\n"
            f"  EXERCISE_MAX_DROP   = {self.EXERCISE_MAX_PEAK_DROP:.0f} px/s  [tightened from 1200]\n"
            f"  STREAM_FPS (fixed dt)= {self.STREAM_FPS}\n"
            f"  overhead_override   = {self.overhead_camera_override}"
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
                "GROUND_CONFIRM_RATIO":      0.45,  # raised from 0.30 — must be substantially horizontal
                "MIN_FALL_ANGLE_PATH_B":     60,
                "MIN_TILT_FRAMES_PATH_B":    15,    # raised from 8 — bends/leans resolve faster than falls
                "HOT_DESCENT_FRAMES":        5,
                "EXERCISE_MIN_RATIO_FRAMES": 10,
            }
        return {
            "IS_UPRIGHT_RATIO":          0.52,
            "IMPACT_LATCH_CLEAR_RATIO":  0.30,
            "GROUND_CONFIRM_RATIO":      0.50,  # raised from 0.30 — must be substantially horizontal
            "MIN_FALL_ANGLE_PATH_B":     25,
            "MIN_TILT_FRAMES_PATH_B":    12,    # raised from 6 — bends/leans resolve faster than falls
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
        took_long     = elapsed     > self.EXERCISE_SLOW_DESCENT_TIME
        gradual_ratio = frame_count > exercise_min_ratio_frames

        if frame_count > 25 and elapsed > 1.0:
            self._debug(
                f"{person_key} 🪑 SEATED SUPPRESSION "
                f"(frames={frame_count} elapsed={elapsed:.2f}s)"
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
            self.last_seen.pop(k, None)

    def _compute_severity(self, person_key) -> str:
        """
        Derive fall severity for a person from current pipeline state.

        CRITICAL – HIGH alert already fired AND person has remained on the ground
                   for ≥ CRITICAL_GROUND_TIME seconds since the alert.
        HIGH     – Impact latched AND ground confirmation accumulated.
        MEDIUM   – Descent counter at threshold OR impact latched but ground not yet confirmed.
        LOW      – None of the above.
        """
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
        """
        Return the current FallSeverity for a specific tracked person.
        Returns FallSeverity.LOW if the person is not being tracked.
        """
        return self.current_severity.get((cam_id, track_id), FallSeverity.LOW)

    def process_batch(self, scene_state: dict) -> list:
        """
        scene_state: {
            cam_id: {
                "tracked_people": {
                    track_id: {
                        "bbox":       [x1,y1,x2,y2],   # pixel coords
                        "confidence": float,
                        "keypoints":  array[17,2|3],    # xy or xyc
                        # optional:
                        "frame_width":  int,
                        "frame_height": int,
                    }
                }
            }
        }

        Returns list of alert dicts. Each alert now includes a "severity" field:
            {
                "cameraId":   str,
                "trackId":    any,
                "eventType":  "fall",
                "confidence": float,
                "severity":   "HIGH" | "MEDIUM" | "LOW",
            }

        Severity is also updated continuously on self.current_severity so
        callers can poll it between alerts via get_severity().
        """
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

                # ── Allow per-person frame-size override ───────────────────
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
                    self._debug(f"[FallPlugin] SKIP cam={cam_id} id={track_id} — bbox is None")
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                if conf < self.conf_threshold:
                    self._debug(
                        f"[FallPlugin] SKIP cam={cam_id} id={track_id} "
                        f"— conf {conf:.2f} < {self.conf_threshold}"
                    )
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                if kps_raw is None:
                    self._debug(f"[FallPlugin] SKIP cam={cam_id} id={track_id} — keypoints is None")
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                kps = np.array(kps_raw, dtype=float)

                if kps.ndim != 2 or kps.shape[0] < 17:
                    self._debug(
                        f"[FallPlugin] SKIP cam={cam_id} id={track_id} "
                        f"— bad kps shape {kps.shape}"
                    )
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                kps_xy = kps[:, :2]

                # ── Keypoint confidence check ──────────────────────────────
                if kps.shape[1] >= 3:
                    kps_conf      = kps[:, 2]
                    critical_conf = float(np.mean(kps_conf[self.CRITICAL_KPS]))
                    if critical_conf < self.KP_CONF_THRESH:
                        self._debug(
                            f"[FallPlugin] SKIP cam={cam_id} id={track_id} "
                            f"— critical kp conf {critical_conf:.2f} < {self.KP_CONF_THRESH}"
                        )
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue
                else:
                    xs    = kps_xy[:, 0]
                    ys    = kps_xy[:, 1]
                    valid = (xs > 0) & (ys > 0)
                    if int(valid.sum()) < 8:
                        self._debug(
                            f"[FallPlugin] SKIP cam={cam_id} id={track_id} "
                            f"— only {int(valid.sum())} valid kps (need 8)"
                        )
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue

                # ── Ghost / edge / area guards ─────────────────────────────
                x1b, y1b, x2b, y2b = bbox
                cx = (x1b + x2b) / 2
                cy = (y1b + y2b) / 2
                em = self.EDGE_MARGIN

                if cx < em or cx > fw - em or cy < em or cy > fh - em:
                    self._debug(
                        f"[FallPlugin] SKIP cam={cam_id} id={track_id} "
                        f"— frame-edge ghost (cx={cx:.0f} cy={cy:.0f})"
                    )
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                bbox_area = (x2b - x1b) * (y2b - y1b)
                if bbox_area < self.MIN_PERSON_AREA:
                    self._debug(
                        f"[FallPlugin] SKIP cam={cam_id} id={track_id} "
                        f"— bbox too small ({bbox_area:.0f} px²)"
                    )
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

                thresholds                = self._camera_thresholds(cam_id)
                hot_descent_frames        = thresholds["HOT_DESCENT_FRAMES"]
                exercise_min_ratio_frames = thresholds["EXERCISE_MIN_RATIO_FRAMES"]
                is_overhead               = self.camera_mode.get(cam_id, False)
                is_upright                = body_ratio < thresholds["IS_UPRIGHT_RATIO"]

                # ── Hip velocity ───────────────────────────────────────────
                hip_mid_y = float((kps_xy[11][1] + kps_xy[12][1]) / 2.0)
                now_ts    = time.time()
                prev_t    = self.prev_time.get(person_key, now_ts)
                dt        = 1.0 / self.STREAM_FPS
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
                    baseline_locked = self.is_baseline_locked.get(person_key, False)
                    if not baseline_locked:
                        self._debug(
                            f"[FallPlugin] SKIP cam={cam_id} id={track_id} "
                            f"— pre-baseline hip jump ({hip_jump:.1f} px)"
                        )
                        self.prev_hip_y[person_key] = hip_mid_y
                        self.velocity_y[person_key] = 0.0
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue
                    else:
                        self._debug(
                            f"[FallPlugin] LARGE hip drop cam={cam_id} id={track_id} "
                            f"({hip_jump:.1f} px) — processing"
                        )

                raw_drop = (hip_mid_y - prev_hip) / dt
                self.prev_hip_y[person_key] = hip_mid_y

                cap = self._velocity_cap(person_key, hot_descent_frames)
                if raw_drop > cap:
                    self._debug(
                        f"[FallPlugin] CLAMP cam={cam_id} id={track_id} "
                        f"raw_drop={raw_drop:.0f} > cap={cap:.0f} — zeroed"
                    )
                    drop = 0.0
                elif raw_drop < 0:
                    drop = 0.0
                else:
                    drop = float(raw_drop)

                self.velocity_y[person_key] = drop

                self._debug(
                    f"[FallPlugin] PHYSICS cam={cam_id} id={track_id} "
                    f"ratio={body_ratio:.2f} angle={torso_angle:.1f} "
                    f"drop={drop:.1f} is_upright={is_upright} "
                    f"descent={self.descent_counter[person_key]} "
                    f"impact={self.impact_detected[person_key]} "
                    f"baseline_locked={self.is_baseline_locked.get(person_key, False)}"
                )

                self._debug(
                    f"[CHECK] descent={self.descent_counter[person_key]} "
                    f"impact={self.impact_detected[person_key]} "
                    f"ground={self.ground_confirm_accum[person_key]:.3f}"
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
                        self._debug(f"[FallPlugin] tilt clock started cam={cam_id} id={track_id}")

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
                        self._debug(f"[FallPlugin] in cooldown cam={cam_id} id={track_id}")
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
                        continue
                else:
                    self.falling_ids.add(person_key)

                # ══════════════════════════════════════════════════════════
                # 5️⃣  IMPACT DETECTION
                # ══════════════════════════════════════════════════════════
                if not self.impact_detected[person_key]:
                    ratio_now   = body_ratio
                    tilt_frames = self.tilt_frame_count[person_key]
                    current_cap = self._velocity_cap(person_key, hot_descent_frames)

                    # Path A — velocity spike
                    if self.IMPACT_VEL_THRESH < drop <= current_cap:
                        path_a_blocked = False

                        if is_overhead:
                            tilt_elapsed_now = now_ts - self.tilt_start_time.get(person_key, now_ts)
                            seated_at_spike  = (ratio_now >= 0.60 and tilt_elapsed_now < 0.5)
                            long_seated_ctx  = self._is_exercise_motion(
                                person_key, exercise_min_ratio_frames
                            )
                            if seated_at_spike or long_seated_ctx:
                                path_a_blocked = True
                                self._debug(
                                    f"[FallPlugin] Path-A BLOCKED (seated) "
                                    f"cam={cam_id} id={track_id}"
                                )

                        if not path_a_blocked:
                            self._latch_impact(person_key, now_ts)
                            self._debug(
                                f"[FallPlugin] 💥 IMPACT-VELOCITY cam={cam_id} id={track_id} "
                                f"drop={drop:.0f} cap={current_cap:.0f}"
                            )

                    # Path B — body horizontal + torso angle + tilt frames + enough descent
                    elif (
                        ratio_now > 0.90                                        # raised from 0.85 — stricter horizontal check
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
                                f"[FallPlugin] 💥 IMPACT-RATIO cam={cam_id} id={track_id} "
                                f"ratio={ratio_now:.2f} angle={torso_angle:.1f} frames={tilt_frames}"
                            )

                # ══════════════════════════════════════════════════════════
                # 6️⃣  GROUND CONFIRMATION
                # ══════════════════════════════════════════════════════════
                if self.impact_detected[person_key]:

                    inertia_elapsed = now_ts - self.inertia_timer.get(person_key, now_ts)
                    if (
                        body_ratio < thresholds["IMPACT_LATCH_CLEAR_RATIO"]
                        and inertia_elapsed > self.INERTIA_TIME
                    ):
                        self.impact_detected[person_key]      = False
                        self.descent_counter[person_key]      = 0
                        self.ground_confirm_accum[person_key] = 0.0
                        self._debug(
                            f"[FallPlugin] stood up, impact cleared "
                            f"(ratio={body_ratio:.2f}) cam={cam_id} id={track_id}"
                        )
                        self.alert_time.pop(person_key, None)
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue

                    if body_ratio > thresholds["GROUND_CONFIRM_RATIO"]:
                        # Hip-rising guard: if hip is moving upward the person is
                        # standing back up — reset accumulation instead of counting
                        hip_rising = (hip_mid_y < prev_hip - 5)
                        if hip_rising:
                            self.ground_confirm_accum[person_key] = 0.0
                            self._debug(
                                f"[FallPlugin] hip rising, ground accum reset "
                                f"cam={cam_id} id={track_id}"
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

                    # ── Update severity before checking confirmation ────────
                    self.current_severity[person_key] = self._compute_severity(person_key)

                    if self.ground_confirm_accum[person_key] >= self.GROUND_CONFIRM_TIME:
                        last_alert = self.last_alert_ts.get(person_key, 0.0)
                        if (now_ts - last_alert) >= self.RECOVERY_COOLDOWN:
                            severity = FallSeverity.HIGH
                            self.current_severity[person_key] = severity
                            self.alert_time[person_key]       = now_ts  # start CRITICAL escalation clock
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

                # ── CRITICAL escalation: person still on ground ≥ CRITICAL_GROUND_TIME ──
                alert_ts = self.alert_time.get(person_key)
                if alert_ts is not None and not self.impact_detected[person_key]:
                    time_on_ground = now_ts - alert_ts
                    if time_on_ground >= self.CRITICAL_GROUND_TIME:
                        # Only fire once per RECOVERY_COOLDOWN window
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

                # ── Severity update for non-impact / mid-pipeline states ───
                self.current_severity[person_key] = self._compute_severity(person_key)

                # ── Quick-recovery: very upright + small angle ─────────────
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