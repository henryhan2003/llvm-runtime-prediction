"""Resident-memory capacity load, deliberately not a continuous bandwidth stressor."""
from __future__ import annotations

import argparse
import json
import mmap
import os
from pathlib import Path
import time


def hold_memory(size, duration, ready):
    deadline = time.monotonic() + duration
    with mmap.mmap(-1, size) as memory:
        page = mmap.PAGESIZE
        for offset in range(0, size, page):
            memory[offset] = 1
            if offset % (64 * 1024 ** 2) == 0 and time.monotonic() >= deadline:
                raise TimeoutError("Timed out while allocating/touching memory")
        path = Path(ready)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"pid": os.getpid(), "resident_pages_touched": (size + page - 1) // page,
            "allocation_bytes": size, "ready_at": time.time()}), encoding="utf-8")
        temporary.replace(path)
        cursor = 0
        while time.monotonic() < deadline:
            # Spread sparse accesses over the allocation without streaming through every byte.
            stride = max(page, size // 2048 // page * page)
            for i in range(min(2048, max(1, size // page))):
                offset = (cursor + i * stride) % size
                memory[offset] = (memory[offset] + 1) % 256
            cursor = (cursor + page) % size
            time.sleep(min(1, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    if args.bytes <= 0 or args.timeout <= 0:
        parser.error("Positive byte count and timeout are required")
    hold_memory(args.bytes, args.timeout, args.ready)


if __name__ == "__main__":
    main()
