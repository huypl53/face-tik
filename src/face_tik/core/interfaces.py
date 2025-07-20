"""
FaceRecognizer Interface
-----------------------
Defines the contract for all face recognition engines. Concrete implementations must inherit from this class.
This interface enables a pluggable architecture using the Strategy Design Pattern.
"""

import abc
import numpy as np

from face_tik.schemas.face import FaceDatabase, RecognitionResult


class FaceRecognizer(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config=None):
        """Initialize the recognizer with optional configuration."""
        pass

    @abc.abstractmethod
    def build_database(self, known_faces_dir: str) -> dict:
        """
        Process the known_faces directory, generate embeddings for each person, and return a database structure:
        { 'person_name': [embedding1, embedding2, ...], ... }
        """
        pass

    @abc.abstractmethod
    def recognize_faces(
        self, image: np.ndarray, face_database: FaceDatabase, top_k: int = 2
    ) -> list[RecognitionResult]:
        """
        Recognize faces in a single image frame.
        Args:
            image: np.ndarray, the image/frame to process
            face_database: dict, the pre-built face embeddings database
            top_k: top k recognized faces
        Returns:
            List of dicts, each with:
                - name: str (person name or 'Unknown')
                - bbox: [x1, y1, x2, y2]
                - confidence: float (recognition score if available)
        """
        pass

