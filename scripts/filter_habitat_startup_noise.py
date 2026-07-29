#!/usr/bin/env python3
"""Filter known harmless Habitat startup lines without buffering progress."""

import re
import sys


NOISY_PLUGIN_LINE = re.compile(
    rb"^PluginManager::Manager: duplicate static plugin .* ignoring$"
)


def should_drop(line: bytes) -> bool:
    return NOISY_PLUGIN_LINE.fullmatch(line) is not None


def main() -> None:
    source = sys.stdin.buffer
    destination = sys.stdout.buffer
    pending = b""

    while True:
        chunk = source.read1(65536)
        if not chunk:
            break
        pending += chunk

        while True:
            newline = pending.find(b"\n")
            carriage_return = pending.find(b"\r")
            separators = [
                position
                for position in (newline, carriage_return)
                if position >= 0
            ]
            if not separators:
                break

            position = min(separators)
            line = pending[:position]
            separator = pending[position : position + 1]
            pending = pending[position + 1 :]
            if not should_drop(line):
                destination.write(line + separator)

        destination.flush()

    if pending and not should_drop(pending):
        destination.write(pending)
    destination.flush()


if __name__ == "__main__":
    main()
