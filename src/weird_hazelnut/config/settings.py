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

    # Override MinIO public endpoint with environment variables if present
    env_minio_public_endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT")
    if env_minio_public_endpoint:
        if "data_layer" in cfg and "object_storage" in cfg["data_layer"]:
            cfg["data_layer"]["object_storage"]["public_endpoint"] = env_minio_public_endpoint

    # Override Loki Cloud Logging with environment variables
    if "cloud_logging" not in cfg:
        cfg["cloud_logging"] = {}

    env_loki_enabled = os.getenv("LOKI_ENABLED")
    if env_loki_enabled is not None:
        cfg["cloud_logging"]["enabled"] = env_loki_enabled.lower() in ("true", "1", "yes")

    env_loki_url = os.getenv("LOKI_URL")
    if env_loki_url:
        cfg["cloud_logging"]["loki_url"] = env_loki_url

    env_loki_user = os.getenv("LOKI_USER")
    if env_loki_user:
        cfg["cloud_logging"]["loki_user"] = env_loki_user

    env_loki_password = os.getenv("LOKI_PASSWORD")
    if env_loki_password:
        cfg["cloud_logging"]["loki_password"] = env_loki_password

    return cfg
