"""Tiny stopwatch: records how long each pipeline stage takes, so we optimize what is really slow."""
import time
from collections import defaultdict
from contextlib import contextmanager

TIMINGS = defaultdict(list)


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        TIMINGS[name].append(time.perf_counter() - start)


def summary_rows():
    """(stage, calls, total seconds, average seconds per call), slowest total first."""
    rows = [(name, len(v), sum(v), sum(v) / len(v)) for name, v in TIMINGS.items()]
    return sorted(rows, key=lambda r: -r[2])