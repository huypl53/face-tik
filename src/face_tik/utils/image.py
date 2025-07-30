from pathlib import Path
from typing import List


def get_image_in_dir(directory: str | Path) -> List[Path]:
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

    d = Path(directory)
    files: List[Path] = []
    for f in d.glob("*"):
        if f.suffix.lower() in image_extensions:
            files.append(f)

    return files
