import pytest
from loguru import logger
from _pytest.logging import LogCaptureFixture


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level="info",
        # filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=True,  # Set to False if no multiprocessing
    )
    yield caplog
    logger.remove(handler_id)
