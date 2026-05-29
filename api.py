import uvicorn

from src.weird_hazelnut.api.app import app
from src.weird_hazelnut.config import load_config


if __name__ == "__main__":
    config = load_config()
    server_cfg = config.get("server", {})
    uvicorn.run(
        app,
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8000),
    )
