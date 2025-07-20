from loguru import logger
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_PATH = os.path.join(LOG_DIR, "recognition_log.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()
logger.add(
    LOG_PATH,
    rotation="10 MB",
    retention=5,
    colorize=False,
    level="WARNING",
    format="{time} | {level} | {file.path}:{line} | {message}",
)
logger.add(
    "stdout",
    level="INFO",
    format="{time} | {level} | {file.path}:{line} | {message}",
    colorize=True,
)

