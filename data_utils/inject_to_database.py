# This script is based on an original implementation by True Price.
# https://www.cnblogs.com/li-minghao/p/11865794.html
# Created by liminghao
import sys
import numpy as np
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_POSE_PRIORS_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

_PRIOR_Q_COLUMN_CANDIDATES = (
    ("prior_qw", "qvec_prior_w"),
    ("prior_qx", "qvec_prior_x"),
    ("prior_qy", "qvec_prior_y"),
    ("prior_qz", "qvec_prior_z"),
)
_PRIOR_T_COLUMN_CANDIDATES = (
    ("prior_tx", "tvec_prior_x"),
    ("prior_ty", "tvec_prior_y"),
    ("prior_tz", "tvec_prior_z"),
)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)
        self._image_prior_spec = None

    def _get_image_prior_spec(self):
        if self._image_prior_spec is None:
            cursor = self.execute("PRAGMA table_info(images)")
            column_names = {row[1] for row in cursor.fetchall()}

            def _select_candidates(candidate_groups):
                selected = []
                for group in candidate_groups:
                    for candidate in group:
                        if candidate in column_names:
                            selected.append(candidate)
                            break
                return selected

            prior_q_columns = _select_candidates(_PRIOR_Q_COLUMN_CANDIDATES)
            prior_t_columns = _select_candidates(_PRIOR_T_COLUMN_CANDIDATES)

            if len(prior_q_columns) == 4 and len(prior_t_columns) == 3:
                self._image_prior_spec = {
                    "storage": "images",
                    "q_columns": tuple(prior_q_columns),
                    "t_columns": tuple(prior_t_columns),
                }
            else:
                pose_priors_exists = bool(
                    self.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='pose_priors'"
                    ).fetchone()
                )
                if pose_priors_exists:
                    self._image_prior_spec = {"storage": "pose_priors"}
                else:
                    known_columns = {
                        candidate
                        for group in (
                            _PRIOR_Q_COLUMN_CANDIDATES + _PRIOR_T_COLUMN_CANDIDATES
                        )
                        for candidate in group
                    }
                    missing = sorted(known_columns - column_names)
                    raise RuntimeError(
                        "Unexpected COLMAP images schema: missing known prior columns "
                        "and pose_priors table (absent: {})".format(
                            ", ".join(missing)
                        )
                    )
        return self._image_prior_spec

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id))
        return cursor.lastrowid

    def update_image(self, IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID):
        prior_spec = self._get_image_prior_spec()
        if prior_spec["storage"] == "images":
            prior_q_columns = prior_spec["q_columns"]
            prior_t_columns = prior_spec["t_columns"]
            column_updates = [f"{col}=?" for col in prior_q_columns + prior_t_columns]
            column_updates.append("camera_id=?")
            sql = f"UPDATE images SET {', '.join(column_updates)} WHERE image_id=?"
            values = (QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_ID)
            cursor = self.execute(sql, values)
            return cursor.lastrowid

        cursor = self.execute(
            "UPDATE images SET camera_id=? WHERE image_id=?",
            (CAMERA_ID, IMAGE_ID),
        )
        position = np.asarray([TX, TY, TZ], dtype=np.float64)
        covariance = np.full((3, 3), np.nan, dtype=np.float64)
        self.execute(
            "INSERT OR REPLACE INTO pose_priors "
            "(image_id, position, coordinate_system, position_covariance) "
            "VALUES (?, ?, ?, ?)",
            (
                IMAGE_ID,
                array_to_blob(position),
                -1,
                array_to_blob(covariance),
            ),
        )
        return cursor.lastrowid

def camTodatabase(txtfile, dbfile):

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    # Open the database.
    db = COLMAPDatabase.connect(dbfile)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] # SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0, len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()

def imgTodatabase(txtfile, dbfile):
    # Open the database.
    db = COLMAPDatabase.connect(dbfile)
    with open(txtfile, "r") as images:
        lines = images.readlines()
        for i in range(0, len(lines)):
            image_metas = lines[
                i
            ].split()  # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            if len(image_metas) > 0:
                db.update_image(
                    IMAGE_ID=int(image_metas[0]),
                    QW=float(image_metas[1]),
                    QX=float(image_metas[2]),
                    QY=float(image_metas[3]),
                    QZ=float(image_metas[4]),
                    TX=float(image_metas[5]),
                    TY=float(image_metas[6]),
                    TZ=float(image_metas[7]),
                    CAMERA_ID=int(image_metas[8]),
                )
    # Commit the data to the file.
    db.commit()
    # Close database.db.
    db.close()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", default="")
    args = parser.parse_args()

    camTodatabase(txtfile=f"{args.input_path}/colmap/sparse/origin/cameras.txt", 
                  dbfile=f"{args.input_path}/colmap/database.db")

    imgTodatabase(txtfile=f"{args.input_path}/colmap/sparse/origin/images.txt",
                  dbfile=f"{args.input_path}/colmap/database.db")
