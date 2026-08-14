#!/usr/bin/env python3
"""Filter known harmless Habitat startup lines without buffering progress."""

import re
import sys


NOISY_PLUGIN_FRAGMENT = re.compile(
    rb"PluginManager::Manager: duplicate static plugin [^,\r\n]+, ignoring"
)
TQDM_PROGRESS_PREFIX = re.compile(rb"^\s*\d+%\|")


def filter_segment(segment: bytes, separator: bytes = b"") -> bytes:
    """Remove harmless plugin messages from one CR/LF-delimited segment."""
    cleaned, removed = NOISY_PLUGIN_FRAGMENT.subn(b"", segment)
    if removed == 0:
        return segment + separator
    if not cleaned:
        return b""

    # Magnum can write its newline immediately after tqdm's current progress
    # text. Keep that text replaceable instead of turning it into a permanent
    # log line.
    if separator == b"\n" and TQDM_PROGRESS_PREFIX.match(cleaned):
        return cleaned + b"\r"
    return cleaned + separator


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
            destination.write(filter_segment(line, separator))

        destination.flush()

    if pending:
        destination.write(filter_segment(pending))
    destination.flush()


if __name__ == "__main__":
    main()
