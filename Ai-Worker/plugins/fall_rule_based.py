import time

import numpy as np

from collections import defaultdict, deque


class FallSeverity:
    HIGH = "HIGH"
    LOW  = "LOW"


class RuleBasedFallPlugin:

    REF_WIDTH = 1920

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        frame_width: int = 1920,
        frame_height: int = 1080,
        debug: bool = False,
    ):
        print("⚡ Initializing Fall Detection Plugin (Posture-Based)...")

        self.conf_threshold = confidence_threshold
        self.debug          = debug
        self.frame_width    = frame_width
        self.frame_height   = frame_height

        res_scale = frame_width / self.REF_WIDTH

        self.RECOVERY_COOLDOWN = 5.0
        self.KP_CONF_THRESH    = 0.30
        self.CRITICAL_KPS      = [5, 6, 11, 12, 13, 14, 15, 16]
        self.EDGE_MARGIN       = max(15, int(40 * res_scale))
        self.MIN_PERSON_AREA   = max(500, int(4000 * res_scale ** 2))

        # ── Posture thresholds ──────────────────────────────────────────────
        self.FALLEN_RATIO_THRESH = 1.0   # body wider than tall → likely fallen
        self.FALLEN_ANGLE_THRESH = 45.0  # torso angle from vertical (degrees)

        # ── Fix 1: Hip height threshold ─────────────────────────────────────
        # If hip midpoint is in the bottom N% of the frame, person is likely
        # sitting on the floor (intentional), not fallen.
        # 0.75 means hips must be below 75% of frame height to be flagged as
        # floor-sit. A fallen person usually has hips in mid-frame.
        self.HIP_FLOOR_SIT_THRESH = 0.75  # normalised Y (0=top, 1=bottom)

        # ── Fix 2: Velocity / suddenness ────────────────────────────────────
        # Tracks the Y centre of the bbox over recent frames.
        # A real fall = rapid downward movement. Sitting down = slow and gradual.
        self.FALL_VELOCITY_THRESH  = 15   # pixels/frame drop to be "sudden"
        self.VELOCITY_HISTORY_LEN  = 5    # frames to smooth velocity over
        self.position_history      = defaultdict(lambda: deque(maxlen=self.VELOCITY_HISTORY_LEN))

        # ── Fix 3: Duration guard ────────────────────────────────────────────
        # If a person has been in "fallen posture" for > this many seconds
        # BEFORE the alert fires, they likely sat down intentionally.
        # A real fall resolves quickly or stays stationary after a sudden drop.
        self.MAX_SETTLE_DURATION   = 8.0  # seconds
        self.fallen_posture_start  = {}   # person_key → timestamp when fallen posture first seen

        # How many consecutive frames the person must look fallen before alert
        self.FALLEN_FRAME_CONFIRM = 3

        self.last_alert_ts    = {}
        self.cooldown_timer   = {}
        self.last_seen        = {}
        self.current_severity = {}
        self.inferred_sizes   = {}
        self.fallen_frame_count = defaultdict(int)

        # print(f"  Default frame size        : {frame_width}x{frame_height}")
        # print(f"  Edge margin               : {self.EDGE_MARGIN}px")
        # print(f"  Min person area           : {self.MIN_PERSON_AREA}px²")
        # print(f"  Fallen body ratio thresh  : > {self.FALLEN_RATIO_THRESH}")
        # print(f"  Fallen torso angle        : > {self.FALLEN_ANGLE_THRESH}°")
        # print(f"  Confirm frames needed     : {self.FALLEN_FRAME_CONFIRM}")
        # print(f"  Hip floor-sit threshold   : > {self.HIP_FLOOR_SIT_THRESH} (normalised Y)")
        # print(f"  Fall velocity threshold   : > {self.FALL_VELOCITY_THRESH} px/frame")
        # print(f"  Max settle duration       : {self.MAX_SETTLE_DURATION}s")

    def _debug(self, msg):
        if self.debug:
            print(msg, flush=True)

    def _log(self, cam_id, track_id, msg):
        pass
        #print(f"  [cam={cam_id} | track={track_id}] {msg}", flush=True)

    def _cleanup_stale_tracks(self, now_ts, ttl=2.0):
        stale = [k for k, t in self.last_seen.items() if (now_ts - t) > ttl]
        for k in stale:
            self.last_alert_ts.pop(k, None)
            self.cooldown_timer.pop(k, None)
            self.current_severity.pop(k, None)
            self.fallen_frame_count.pop(k, None)
            self.fallen_posture_start.pop(k, None)
            self.position_history.pop(k, None)
            self.last_seen.pop(k, None)

    def _resolve_frame_size(self, cam_id: str, person: dict, bbox) -> tuple:
        fw = int(person.get("frame_width", 0))
        fh = int(person.get("frame_height", 0))
        if fw > 100 and fh > 100:
            x1, y1, x2, y2 = bbox
            if x2 <= fw and y2 <= fh:
                self.inferred_sizes[cam_id] = (fw, fh)
                return fw, fh

        if cam_id in self.inferred_sizes:
            return self.inferred_sizes[cam_id]

        x1, y1, x2, y2 = bbox
        COMMON_WIDTHS = [640, 960, 1280, 1920, 2560, 3840]
        inferred_w = next((w for w in COMMON_WIDTHS if w >= x2), int(x2 * 1.1))
        inferred_h = int(inferred_w * 9 / 16)
        self.inferred_sizes[cam_id] = (inferred_w, inferred_h)
        return inferred_w, inferred_h

    @staticmethod
    def _compute_torso_angle(kps_xy: np.ndarray) -> float:
        """
        Angle of the torso from vertical (0° = perfectly upright, 90° = horizontal).
        Uses shoulder midpoint → hip midpoint vector.
        """
        shoulder_mid = (kps_xy[5] + kps_xy[6]) / 2
        hip_mid      = (kps_xy[11] + kps_xy[12]) / 2
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        return float(abs(np.degrees(np.arctan2(dx, dy))))

    @staticmethod
    def _compute_body_ratio(kps_xy: np.ndarray) -> float:
        """
        Width / height of the bounding box around all keypoints.
        < 0.5  → tall upright person
        0.5–1  → bent / crouching
        > 1.0  → horizontal / fallen
        """
        xs     = kps_xy[:, 0]
        ys     = kps_xy[:, 1]
        width  = float(np.max(xs) - np.min(xs))
        height = float(np.max(ys) - np.min(ys))
        if height <= 0:
            return 0.0
        return min(width / height, 3.0)

    def _is_fallen_posture(self, kps_xy: np.ndarray, cam_id: str, track_id) -> tuple[bool, float, float]:
        body_ratio  = self._compute_body_ratio(kps_xy)
        torso_angle = self._compute_torso_angle(kps_xy)

        ratio_ok  = body_ratio  > self.FALLEN_RATIO_THRESH
        angle_ok  = torso_angle > self.FALLEN_ANGLE_THRESH
        is_fallen = ratio_ok and angle_ok

        self._log(
            cam_id, track_id,
            f"📐 Posture | ratio={body_ratio:.2f} (need>{self.FALLEN_RATIO_THRESH}) "
            f"angle={torso_angle:.1f}° (need>{self.FALLEN_ANGLE_THRESH}°) "
            f"→ {'🔴 FALLEN' if is_fallen else '🟢 upright/bending'}"
        )
        return is_fallen, body_ratio, torso_angle

    # ── FIX 1: Hip height guard ──────────────────────────────────────────────
    def _is_floor_sitting(self, kps_xy: np.ndarray, fh: int, cam_id: str, track_id) -> bool:
        """
        Returns True if the person's hips are very close to the bottom of the
        frame, which indicates intentional floor-sitting rather than a fall.

        A fallen person typically has their hips in the mid-frame area.
        Someone sitting cross-legged or slumped on the floor has hips very low.
        """
        hip_mid_y      = (kps_xy[11][1] + kps_xy[12][1]) / 2
        hip_normalised = hip_mid_y / fh if fh > 0 else 0.0
        is_sitting     = hip_normalised > self.HIP_FLOOR_SIT_THRESH

        self._log(
            cam_id, track_id,
            f"🪑 Hip height | normalised_y={hip_normalised:.2f} "
            f"(floor-sit threshold={self.HIP_FLOOR_SIT_THRESH}) "
            f"→ {'🟡 floor-sitting (skip)' if is_sitting else '✅ not floor-sit'}"
        )
        return is_sitting

    # ── FIX 2: Velocity / suddenness guard ──────────────────────────────────
    def _is_sudden_fall(self, person_key, cy: float, cam_id: str, track_id) -> bool:
        """
        Returns True if the centre-Y of the bbox has dropped rapidly (sudden
        downward motion), which is the signature of a real fall.

        Sitting down intentionally produces a slow, gradual movement that stays
        well below this velocity threshold.
        """
        history = self.position_history[person_key]
        history.append(cy)

        if len(history) < 2:
            # Not enough history yet — treat as sudden to avoid missing real falls
            self._log(cam_id, track_id, "⚡ Velocity | not enough history — treating as sudden")
            return True

        # Average velocity over the last N frames (pixels/frame, positive = downward)
        diffs    = [history[i] - history[i - 1] for i in range(1, len(history))]
        avg_vel  = float(np.mean(diffs))
        is_sudden = avg_vel > self.FALL_VELOCITY_THRESH

        self._log(
            cam_id, track_id,
            f"⚡ Velocity | avg_drop={avg_vel:.1f}px/frame "
            f"(sudden threshold={self.FALL_VELOCITY_THRESH}) "
            f"→ {'🔴 sudden drop' if is_sudden else '🟡 slow movement (skip)'}"
        )
        return is_sudden

    # ── FIX 3: Duration / settle guard ──────────────────────────────────────
    def _is_settled_too_long(self, person_key, now_ts: float, cam_id: str, track_id) -> bool:
        """
        Returns True if the person has been in a fallen-like posture for too
        long BEFORE the alert fires.

        Real falls either resolve quickly (person gets up) or remain frozen
        after a sudden drop. Someone who sat down gradually will have been in
        this posture for many seconds before the alert threshold is reached.
        """
        start = self.fallen_posture_start.get(person_key)
        if start is None:
            self.fallen_posture_start[person_key] = now_ts
            return False

        duration = now_ts - start
        too_long = duration > self.MAX_SETTLE_DURATION

        self._log(
            cam_id, track_id,
            f"⏱️  Duration | posture_held={duration:.1f}s "
            f"(max={self.MAX_SETTLE_DURATION}s) "
            f"→ {'🟡 settled too long (skip)' if too_long else '✅ within limit'}"
        )
        return too_long

    def process_batch(self, scene_state: dict) -> list:
        alerts     = []
        now_global = time.time()

        total_cameras = len(scene_state)
        total_people  = sum(len(d.get("tracked_people", {})) for d in scene_state.values())
        #print(f"\n📦 Batch | cameras={total_cameras} | people={total_people}", flush=True)

        for cam_id, data in scene_state.items():
            tracked = data.get("tracked_people", {})

            for track_id, person in tracked.items():
                person_key = (cam_id, track_id)
                self.last_seen[person_key] = now_global

                bbox    = person.get("bbox")
                conf    = float(person.get("confidence", 0.0))
                kps_raw = person.get("keypoints")

                self._log(cam_id, track_id, f"🔍 conf={conf:.2f} | bbox={bbox}")

                # ── Basic validity checks ──────────────────────────────────
                if bbox is None or kps_raw is None:
                    self._log(cam_id, track_id, "⏭️  SKIP — missing bbox or keypoints")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                if conf < self.conf_threshold:
                    self._log(cam_id, track_id, f"⏭️  SKIP — conf {conf:.2f} < {self.conf_threshold:.2f}")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                kps = np.array(kps_raw, dtype=float)
                if kps.ndim != 2 or kps.shape[0] < 17:
                    self._log(cam_id, track_id, f"⏭️  SKIP — bad keypoints shape {kps.shape}")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                kps_xy = kps[:, :2]

                # ── Keypoint confidence ────────────────────────────────────
                if kps.shape[1] >= 3:
                    critical_kp_conf = float(np.mean(kps[self.CRITICAL_KPS, 2]))
                    if critical_kp_conf < self.KP_CONF_THRESH:
                        self._log(cam_id, track_id, f"⏭️  SKIP — kp conf {critical_kp_conf:.2f} < {self.KP_CONF_THRESH:.2f}")
                        self.current_severity[person_key] = FallSeverity.LOW
                        self.fallen_frame_count[person_key] = 0
                        self.fallen_posture_start.pop(person_key, None)
                        continue
                else:
                    xs, ys = kps_xy[:, 0], kps_xy[:, 1]
                    valid_kps = int(((xs > 0) & (ys > 0)).sum())
                    if valid_kps < 8:
                        self._log(cam_id, track_id, f"⏭️  SKIP — only {valid_kps}/17 valid kps")
                        self.current_severity[person_key] = FallSeverity.LOW
                        self.fallen_frame_count[person_key] = 0
                        self.fallen_posture_start.pop(person_key, None)
                        continue

                # ── Frame size + edge/area checks ──────────────────────────
                x1, y1, x2, y2 = bbox
                fw, fh = self._resolve_frame_size(cam_id, person, bbox)
                actual_scale = fw / self.REF_WIDTH
                em   = max(15, int(40 * actual_scale))
                area = (x2 - x1) * (y2 - y1)
                cx   = (x1 + x2) / 2
                cy   = (y1 + y2) / 2

                if cx < em or cx > fw - em or cy < em or cy > fh - em:
                    self._log(cam_id, track_id, "⏭️  SKIP — too close to edge")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                min_area = max(500, int(4000 * actual_scale ** 2))
                if area < min_area:
                    self._log(cam_id, track_id, f"⏭️  SKIP — area {area:.0f} < {min_area}")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                # ── Cooldown check ─────────────────────────────────────────
                now_ts     = time.time()
                cool_start = self.cooldown_timer.get(person_key)
                if cool_start is not None:
                    remaining = self.RECOVERY_COOLDOWN - (now_ts - cool_start)
                    if remaining > 0:
                        self._log(cam_id, track_id, f"⏳ Cooldown — {remaining:.1f}s left")
                        self.current_severity[person_key] = FallSeverity.LOW
                        continue
                    else:
                        self.cooldown_timer.pop(person_key, None)

                # ── Posture check ──────────────────────────────────────────
                is_fallen, body_ratio, torso_angle = self._is_fallen_posture(kps_xy, cam_id, track_id)

                if is_fallen:
                    self.fallen_frame_count[person_key] += 1
                    self._log(
                        cam_id, track_id,
                        f"⚠️  Fallen posture confirmed {self.fallen_frame_count[person_key]}"
                        f"/{self.FALLEN_FRAME_CONFIRM} frames"
                    )
                else:
                    if self.fallen_frame_count[person_key] > 0:
                        self._log(cam_id, track_id, "🟢 Posture recovered — resetting fallen counter")
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    self.position_history[person_key].clear()
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                # ── Need N consecutive fallen frames before alerting ────────
                if self.fallen_frame_count[person_key] < self.FALLEN_FRAME_CONFIRM:
                    self.current_severity[person_key] = FallSeverity.LOW
                    continue

                # ══ FALSE POSITIVE GUARDS (all 3 must pass) ════════════════

                # Guard 1 — Hip height: skip if hips are very low (floor-sitting)
                if self._is_floor_sitting(kps_xy, fh, cam_id, track_id):
                    self._log(cam_id, track_id, "🟡 SKIP — floor-sitting posture detected")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                # Guard 2 — Velocity: skip if the drop was gradual (not a real fall)
                if not self._is_sudden_fall(person_key, cy, cam_id, track_id):
                    self._log(cam_id, track_id, "🟡 SKIP — movement too slow to be a fall")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                # Guard 3 — Duration: skip if person was in "fallen" posture too long before alert
                if self._is_settled_too_long(person_key, now_ts, cam_id, track_id):
                    self._log(cam_id, track_id, "🟡 SKIP — posture held too long before alert (slow sit-down)")
                    self.current_severity[person_key] = FallSeverity.LOW
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    continue

                # ── Fire alert ─────────────────────────────────────────────
                last_alert = self.last_alert_ts.get(person_key, 0.0)
                if (now_ts - last_alert) >= self.RECOVERY_COOLDOWN:
                    self.current_severity[person_key] = FallSeverity.HIGH
                    print(
                        f"\n🚨 FALL DETECTED | cam={cam_id} | track={track_id} | "
                        f"conf={conf:.2f} | ratio={body_ratio:.2f} | angle={torso_angle:.1f}° | "
                        f"severity={FallSeverity.HIGH}",
                        flush=True,
                    )
                    alerts.append({
                        "cameraId":    cam_id,
                        "trackId":     track_id,
                        "eventType":   "fall",
                        "confidence":  conf,
                        "severity":    FallSeverity.HIGH,
                        "body_ratio":  round(body_ratio, 3),
                        "torso_angle": round(torso_angle, 1),
                    })
                    self.last_alert_ts[person_key]  = now_ts
                    self.cooldown_timer[person_key] = now_ts
                    self.fallen_frame_count[person_key] = 0
                    self.fallen_posture_start.pop(person_key, None)
                    self.position_history[person_key].clear()

        self._cleanup_stale_tracks(now_global)
        #print(f"  📊 Done | alerts={len(alerts)}\n", flush=True)
        return alerts

    def get_severity(self, cam_id: str, track_id) -> str:
        return self.current_severity.get((cam_id, track_id), FallSeverity.LOW)