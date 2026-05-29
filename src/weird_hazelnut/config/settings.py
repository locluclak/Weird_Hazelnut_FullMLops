import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load YAML configuration.

    If no path is supplied, CONFIG_PATH is used, falling back to config.yaml.
    """
    path = Path(config_path or os.getenv("CONFIG_PATH", "config.yaml"))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

