import pytest
from pathlib import Path
import cv2

from face_tik.recognizers.deepface import DeepFace
from face_tik.schemas.face import FaceDatabase

from loguru import logger


class TestRecognizers:

    @pytest.fixture
    def known_face_dir(self):
        return Path(__file__).parent.parent / "samples" / "known_people"

    @pytest.fixture
    def face_to_verify(self):
        return (
            Path(__file__).parent.parent
            / "samples"
            / "random_people"
            / "trump_860x394.jpg"
        )

    @pytest.fixture
    def face_db(self, known_face_dir: Path) -> FaceDatabase:
        return FaceDatabase(face_dir=known_face_dir)

    def test_deepface_infer_single(self, face_to_verify: Path, face_db: FaceDatabase):
        assert face_to_verify.is_file(), "No image found at {}".format(face_to_verify)
        deep_face_recog = DeepFace()
        input_face = cv2.imread(str(face_to_verify))
        logger.info(f"Read {face_to_verify}... done!")
        if input_face is None:
            raise FileExistsError("Reading image at {} failed".format(face_to_verify))
        # input_face = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)

        faces = deep_face_recog.recognize_faces(input_face, face_db)
        logger.info(f"Found {len(faces)} faces\n{faces}")
