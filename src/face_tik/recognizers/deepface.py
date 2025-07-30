from face_tik.core.interfaces import FaceRecognizer

# from deepface.modules import recognition
from deepface import DeepFace as _DeepFace
from pathlib import Path
import numpy as np
import tempfile
import cv2
import pandas as pd
from typing import cast
from loguru import logger

from face_tik.schemas.face import RecognitionResult, FaceDatabase


class DeepFace(FaceRecognizer):
    def __init__(
        self,
        face_database: FaceDatabase,
    ):
        super().__init__(face_database)

    # TODO: implement DeepFace pickling step
    def build_database(self, face_db: FaceDatabase):
        return super().build_database(face_db)

    @property
    def has_known_faces(self) -> bool:
        return True

    def recognize_faces(
        self, image: np.ndarray, top_k: int = 5
    ) -> list[RecognitionResult]:

        im_file = tempfile.NamedTemporaryFile(suffix=".png", delete=True)
        im_path = im_file.name
        cv2.imwrite(im_path, image)

        face_db_path = self.face_db.face_dir
        dfs = _DeepFace.find(
            img_path=im_path, db_path=str(face_db_path), model_name="Facenet512"
        )

        im_file.close()

        if not isinstance(dfs, list) or not len(dfs):
            return []

        df = pd.concat(dfs).sort_values(by=["distance"])
        logger.info(f"Found {len(df)} matched faces")
        df["identity"] = df["identity"].apply(lambda x: Path(x).stem)
        df: pd.DataFrame = df.iloc[:top_k]
        df["bbox"] = df.apply(
            lambda x: [x["target_x"], x["target_y"], x["target_w"], x["target_h"]],
            axis=1,
        )
        df = cast(pd.DataFrame, df[["identity", "distance", "bbox"]])
        return [RecognitionResult(**r) for r in df.to_dict(orient="records")]
