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
        cfg = yaml.safe_load(f) or {}

    # Override Label Studio config with environment variables if present (Security & Flexibility)
    if "label_studio" not in cfg:
        cfg["label_studio"] = {}
    
    env_api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if env_api_key:
        cfg["label_studio"]["api_key"] = env_api_key
        
    env_url = os.getenv("LABEL_STUDIO_URL")
    if env_url:
        cfg["label_studio"]["url"] = env_url
        
    env_project_id = os.getenv("LABEL_STUDIO_PROJECT_ID")
    if env_project_id:
        try:
            cfg["label_studio"]["project_id"] = int(env_project_id)
        except ValueError:
            pass

    return cfg
