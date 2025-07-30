from face_tik.core.interfaces import FaceRecognizer
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Tuple
from loguru import logger
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from typing import List

from face_tik.schemas.face import RecognitionResult, FaceDatabase
from face_tik.utils.image import get_image_in_dir

_app = FaceAnalysis(
    name="buffalo_l", providers=["CPUExecutionProvider"]
)  # Use 'CUDAExecutionProvider' for GPU
_app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GP


class InsightFace(FaceRecognizer):
    def __init__(
        self,
        face_database: FaceDatabase,
        compare_thresh: float = 0.5,
    ):
        assert compare_thresh <= 1
        self.comp_thresh = compare_thresh
        self.known_faces: Dict[str, Face] = {}
        super().__init__(face_database)

    @property
    def has_known_faces(self) -> bool:
        return any(self.known_faces)

    def build_database(self, face_db: FaceDatabase):
        self.face_db = face_db
        face_dir = Path(self.face_db.face_dir)
        assert face_dir.is_dir()
        known_faces_files = get_image_in_dir(self.face_db.face_dir)

        self.known_faces = {}

        for f in known_faces_files:
            try:
                im = cv2.imread(str(f))
                assert im is not None
                faces = self.detect_faces(im)
                if len(faces) > 1:
                    logger.warning(
                        f"{f} has more than 1 identity face. Found {len(faces)}. Firs face is picked"
                    )
                    faces = [faces[0]]
                elif not (len(faces)):
                    logger.info(f"{f} has no face! Skipped")
                    continue
                self.known_faces[f.stem] = faces[0]
            except Exception as e:
                logger.error(e)

    def recognize_faces(
        self, image: np.ndarray, top_k: int = 5
    ) -> List[RecognitionResult]:

        if not self.has_known_faces:
            logger.warning("Have no known faces. Check face_database first!")
            return []
        faces = self.detect_faces(image)

        if not isinstance(faces, list) or not len(faces):
            return []

        recog_results: List[RecognitionResult] = []
        for face in faces:
            match_results: List[RecognitionResult] = []
            for identity, known_face in self.known_faces.items():
                similarity, _ = self._compare_faces(face, known_face)
                if not similarity > self.comp_thresh:
                    continue
                x1, y1, x2, y2 = known_face["bbox"]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                match_results.append(
                    RecognitionResult(
                        identity=identity, bbox=bbox, distance=1 - similarity
                    )
                )
            recog_results.extend(match_results[:top_k])
        if not len(recog_results):
            logger.info(f"No face recognized")
        return recog_results

    def _compare_faces(self, face1: Face, face2: Face):
        return compare_faces(face1.embedding, face2.embedding, self.comp_thresh)

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        faces: List[Face] = _app.get(image)
        return faces


def compare_faces(
    emb1, emb2, threshold=0.65
) -> Tuple[float, bool]:  # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold
