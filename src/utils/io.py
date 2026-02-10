import json
import os
from typing import Any, Dict, Optional


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Dict[str, Any], *, indent: int = 2) -> None:
    """Write a JSON file to disk."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
