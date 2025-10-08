"""Context manager for timing code blocks with logging and emoji indicators.

Authors: [Your Name]
"""

import time
from contextlib import ContextDecorator

EMOJIS = {
    0: "🔥",  # fire
    1: "🚀",  # rocket
    2: "⏱️",  # stopwatch (not used, but available)
    3: "🕑",  # red clock (closest Unicode, as no red clock exists)
}


class Timing(ContextDecorator):
    def __init__(self, logger, label: str, level: int = 1):
        self.logger = logger
        self.label = label
        self.level = level
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.start_time is None:
            return False
        duration = time.time() - self.start_time
        emoji = EMOJIS.get(self.level, "⏱️")
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        s = duration % 60
        self.logger.info(f"{emoji} {self.label} took {duration:.2f} sec. ({h:02d}:{m:02d}:{s:05.2f})")
        return False
