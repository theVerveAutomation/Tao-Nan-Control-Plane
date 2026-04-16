from config import DATABASE_URL

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
        self.database_url = database_url or DATABASE_URL

    def get_connection(self):
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 is not installed")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not set")
        conn = psycopg2.connect(self.database_url)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        return conn


    @classmethod
    def camera_from_db_row(cls, row):
        if not row:
            return {}

        return {
            # Runtime-required fields
            "url": row.get("url"),
            "id": str(row.get("id")) if row.get("id") is not None else None,
            "organization_id": row.get("organization_id"),
            "name": row.get("name"),
            "status": row.get("status"),
            "detection": row.get("detection"),
            "alert_sound": row.get("alert_sound"),
            "frame_rate": row.get("frame_rate"),
            "resolution": row.get("resolution"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }

    def fetch_all_cameras(self):
        conn = self.get_connection()
        try:
            curs = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            curs.execute(
                """
                SELECT *
                FROM cameras
                ORDER BY created_at ASC;
                """
            )
            rows = curs.fetchall()

            cameras_config = {}
            for row in rows:
                print(f"✅ [DB] Fetched camera {row['id']}: {row['name']}")
                cam_id = str(row.get("id"))
                if not row.get("url"):
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
