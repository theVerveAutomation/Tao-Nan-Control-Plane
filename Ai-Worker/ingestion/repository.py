import os

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.extensions
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    PSYCOPG2_AVAILABLE = False


class PostgresCameraRepository:
    def __init__(self, database_url=None):
        self.database_url = database_url or os.environ.get("DATABASE_URL")

    def get_connection(self):
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 is not installed")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not set")
        conn = psycopg2.connect(self.database_url)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        return conn

    @staticmethod
    def camera_state_from_row(camera_row):
        detection_enabled = bool(camera_row.get("detection", True))
        status = str(camera_row.get("status") or "normal").lower()
        is_active = status == "normal"

        return {
            "is_active": is_active,
            "fall_detection": detection_enabled,
            "tussle_detection": detection_enabled,
        }

    @classmethod
    def camera_from_db_row(cls, row):
        source_url = row.get("stream_url") or row.get("url")
        return {
            # Runtime-required fields
            "url": source_url,
            **cls.camera_state_from_row(row),
            # Full cameras table payload
            "id": str(row.get("id")),
            "organization_id": row.get("organization_id"),
            "name": row.get("name"),
            "location": row.get("location"),
            "status": row.get("status"),
            "detection": row.get("detection"),
            "alert_sound": row.get("alert_sound"),
            "frame_rate": row.get("frame_rate"),
            "resolution": row.get("resolution"),
            "db_url": row.get("url"),
            "is_physical_device": row.get("is_physical_device"),
            "stream_url": row.get("stream_url"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    def fetch_all_cameras(self):
        conn = self.get_connection()
        try:
            curs = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            curs.execute(
                """
                SELECT
                    id,
                    organization_id,
                    name,
                    location,
                    status,
                    detection,
                    alert_sound,
                    frame_rate,
                    resolution,
                    url,
                    is_physical_device,
                    stream_url,
                    created_at,
                    updated_at
                FROM cameras
                ORDER BY created_at ASC;
                """
            )
            rows = curs.fetchall()
            print(f"✅ [DB] Fetched {rows} rows")

            cameras_config = {}
            for row in rows:
                cam_id = str(row.get("id"))
                source_url = row.get("stream_url") or row.get("url")
                if not source_url:
                    continue
                cameras_config[cam_id] = self.camera_from_db_row(row)

            return cameras_config
        finally:
            conn.close()

    def fetch_camera_by_id(self, cam_id):
        conn = self.get_connection()
        try:
            curs = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            curs.execute(
                """
                SELECT
                    id,
                    organization_id,
                    name,
                    location,
                    status,
                    detection,
                    alert_sound,
                    frame_rate,
                    resolution,
                    url,
                    is_physical_device,
                    stream_url,
                    created_at,
                    updated_at
                FROM cameras
                WHERE id = %s
                LIMIT 1;
                """,
                (cam_id,),
            )
            row = curs.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
