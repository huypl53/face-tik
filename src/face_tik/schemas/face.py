from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, Field


class FaceDatabase(BaseModel):
    """
    Configuration object for the face database.
    Attributes:
        face_dir (str): Directory containing known face images, organized by person.
        (You can add more properties as needed, e.g., file extensions, metadata, etc.)
    """

    face_dir: str | Path
    # Add more properties as needed, e.g.:
    # file_extensions: Optional[list[str]] = None
    metadata: dict = Field(default_factory=lambda: {"deepface": {}, "insightface": {}})


class RecognitionResult(BaseModel):
    """
    Represents a single face recognition result.
    Attributes:
        identity (str): The recognized person's name, or 'Unknown'.
        bbox (list[int]): Bounding box as [x1, y1, x2, y2].
        confidence (float): Recognition confidence/score (if available).
        distance (float): Distance of finding target
    """

    identity: str
    bbox: List[int]  # x, y, width, height
    confidence: Optional[float] = None
    distance: Optional[float] = None
