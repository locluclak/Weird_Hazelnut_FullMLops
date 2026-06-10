import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class LabelStudioClient:
    def __init__(self, url: str, api_key: str, project_id: int):
        self.url = url
        self.api_key = api_key
        self.project_id = project_id

        if api_key == "PASTE_YOUR_API_KEY_HERE":
            logger.warning("Label Studio API key not set. SDK calls will fail.")
            self.client = None
        else:
            try:
                from label_studio_sdk.client import LabelStudio

                self.client = LabelStudio(base_url=self.url, api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Label Studio client: %s", e)
                self.client = None

    def create_task(
        self,
        image_url: str,
        score: float,
        metadata: dict | None = None,
    ) -> Optional[int]:
        if not self.client:
            logger.warning("Label Studio client not configured. Skipping task creation.")
            return None

        try:
            task = self.client.tasks.create(
                project=self.project_id,
                data={
                    "image": image_url,
                    "anomaly_score": score,
                    "source": "uncertain_pipeline_routing",
                    **{k: v for k, v in (metadata or {}).items() if v is not None},
                },
            )
            logger.info("Created Label Studio task: %s for image %s", task.id, image_url)
            return task.id
        except Exception as e:
            logger.error("Error creating Label Studio task: %s", e)
            return None

    def update_task_image(self, task_id: int, image_url: str) -> bool:
        if not self.client:
            logger.warning("Label Studio client not configured. Skipping task update.")
            return False

        try:
            task = self.client.tasks.get(str(task_id))
            data = dict(task.data or {})
            data["image"] = image_url
            self.client.tasks.update(str(task_id), data=data)
            logger.info("Updated Label Studio task %s image URL.", task_id)
            return True
        except Exception as e:
            logger.error("Error updating Label Studio task %s: %s", task_id, e)
            return False


def local_file_url(image_path: str) -> str:
    normalized = image_path.replace("\\", "/")
    parts = normalized.split("/data/lake/")
    if len(parts) > 1:
        relative_path = parts[1]
    else:
        relative_path = os.path.basename(image_path)
    return f"http://localhost:8000/static/{relative_path}"
