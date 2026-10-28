"""Stop only our load process group when its owning pipeline dies or times out."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if os.name != "posix" or os.getpgrp() != os.getpid() or not command:
        raise RuntimeError("Supervisor requires its own POSIX session and a command")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for signum in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(signum, stop)
    process = subprocess.Popen(command)
    deadline = time.monotonic() + args.timeout
    while process.poll() is None and not stopped and os.getppid() == args.parent and time.monotonic() < deadline:
        time.sleep(.2)
    code = process.poll()
    # This supervisor is the group leader; descendants inherit this group.
    os.killpg(os.getpgrp(), signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if code is not None:
        # Also kill any descendants left after a nominal stressor exit.
        time.sleep(.2)
    os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == "__main__":
    main()
