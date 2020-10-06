import logging
import time
from contextlib import contextmanager

from .slack import slack_notify


@contextmanager
def timer(name: str, log: bool = True):
    t0 = time.time()
    msg = f"[{name}] start"
    print(msg)
    # slack_notify(msg)

    if log:
        logging.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    print(msg)
    # slack_notify(msg)

    if log:
        logging.info(msg)
