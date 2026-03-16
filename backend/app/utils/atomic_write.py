"""
Atomic file write utility.

Prevents data corruption by writing to a temporary file first,
then atomically renaming it to the target path. If the process
crashes mid-write, the original file remains intact.
"""

import json
import os
import tempfile
from typing import Any


def atomic_json_write(path: str, data: Any, **json_kwargs) -> None:
    """Write a JSON file atomically using write-then-rename.

    Args:
        path: Target file path.
        data: JSON-serializable data.
        **json_kwargs: Extra keyword arguments passed to json.dump
            (e.g. ensure_ascii, indent).
    """
    json_kwargs.setdefault('ensure_ascii', False)
    json_kwargs.setdefault('indent', 2)

    dir_name = os.path.dirname(path) or '.'
    os.makedirs(dir_name, exist_ok=True)

    # Write to a temp file in the same directory (same filesystem = atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, **json_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # Atomic on POSIX; near-atomic on Windows
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
