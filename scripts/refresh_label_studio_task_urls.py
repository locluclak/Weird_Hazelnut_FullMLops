import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sqlalchemy import select

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer
from weird_hazelnut.data.models import ImageRecord, LabelTask
from weird_hazelnut.integrations import LabelStudioClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def refresh_task_urls(task_ids: list[int] | None = None) -> tuple[int, int]:
    config = load_config()
    data_layer = create_data_layer(config)
    if not data_layer:
        raise RuntimeError("data_layer.enabled must be true to refresh Label Studio task URLs")

    ls_cfg = config.get("label_studio", {})
    client = LabelStudioClient(
        url=ls_cfg.get("url"),
        api_key=ls_cfg.get("api_key"),
        project_id=ls_cfg.get("project_id"),
    )

    with data_layer.database.session() as session:
        stmt = select(LabelTask, ImageRecord).join(ImageRecord, ImageRecord.id == LabelTask.image_id)
        if task_ids:
            stmt = stmt.where(LabelTask.label_studio_task_id.in_(task_ids))
        rows = list(session.execute(stmt).all())

    updated = 0
    failed = 0
    for label_task, image in rows:
        image_url = data_layer.presigned_image_url(image.minio_object_key)
        if client.update_task_image(label_task.label_studio_task_id, image_url):
            updated += 1
        else:
            failed += 1

    return updated, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Label Studio task image URLs.")
    parser.add_argument("--task-id", action="append", type=int, help="Refresh only this Label Studio task ID. Can be repeated.")
    args = parser.parse_args()

    updated, failed = refresh_task_urls(args.task_id)
    logger.info("Refreshed %s Label Studio task URLs; %s failed.", updated, failed)


if __name__ == "__main__":
    main()
