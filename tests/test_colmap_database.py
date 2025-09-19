import math
import pathlib
import sqlite3
import struct
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.inject_to_database import COLMAPDatabase


def _create_modern_images_table(db: sqlite3.Connection) -> None:
    db.execute(
        """
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            qvec_prior_w REAL,
            qvec_prior_x REAL,
            qvec_prior_y REAL,
            qvec_prior_z REAL,
            tvec_prior_x REAL,
            tvec_prior_y REAL,
            tvec_prior_z REAL
        )
        """
    )


def _create_pose_prior_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL
        );
        CREATE TABLE pose_priors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            position BLOB,
            coordinate_system INTEGER NOT NULL,
            position_covariance BLOB
        );
        """
    )

def test_update_image_supports_modern_prior_columns():
    db = COLMAPDatabase.connect(":memory:")
    try:
        _create_modern_images_table(db)
        db.execute(
            "INSERT INTO images (image_id, name, camera_id, qvec_prior_w, qvec_prior_x, qvec_prior_y, qvec_prior_z, tvec_prior_x, tvec_prior_y, tvec_prior_z)"
            " VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0)",
            (1, "test.png", 1),
        )

        db.update_image(1, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 2)

        row = db.execute(
            "SELECT qvec_prior_w, qvec_prior_x, qvec_prior_y, qvec_prior_z, tvec_prior_x, tvec_prior_y, tvec_prior_z, camera_id"
            " FROM images WHERE image_id=?",
            (1,),
        ).fetchone()
        assert row == (1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 2)
    finally:
        db.close()


def test_update_image_inserts_pose_prior_rows_when_table_present():
    db = COLMAPDatabase.connect(":memory:")
    try:
        _create_pose_prior_schema(db)
        db.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
            (1, "test.png", 1),
        )

        db.update_image(1, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 2)

        camera_row = db.execute(
            "SELECT camera_id FROM images WHERE image_id=?",
            (1,),
        ).fetchone()
        assert camera_row == (2,)

        pose_prior_row = db.execute(
            "SELECT position, coordinate_system, position_covariance FROM pose_priors WHERE image_id=?",
            (1,),
        ).fetchone()
        assert pose_prior_row is not None

        position = struct.unpack("<3d", pose_prior_row[0])
        assert all(
            math.isclose(value, expected)
            for value, expected in zip(position, (0.1, 0.2, 0.3))
        )
        assert pose_prior_row[1] == -1
        covariance = struct.unpack("<9d", pose_prior_row[2])
        assert all(math.isnan(value) for value in covariance)
    finally:
        db.close()
