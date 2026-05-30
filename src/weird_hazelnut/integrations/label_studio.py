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
        image_path: str,
        score: float,
        metadata: dict | None = None,
    ) -> Optional[int]:
        if not self.client:
            logger.warning("Label Studio client not configured. Skipping task creation.")
            return None

        try:
            relative_path = _to_label_studio_path(image_path)
            task = self.client.tasks.create(
                project=self.project_id,
                data={
                    "image": f"/data/local-files/?d={relative_path}",
                    "anomaly_score": score,
                    "source": "uncertain_pipeline_routing",
                    **{k: v for k, v in (metadata or {}).items() if v is not None},
                },
            )
            logger.info("Created Label Studio task: %s for image %s", task.id, image_path)
            return task.id
        except Exception as e:
            logger.error("Error creating Label Studio task: %s", e)
            return None


def _to_label_studio_path(image_path: str) -> str:
    normalized = image_path.replace("\\", "/")
    parts = normalized.split("/data/lake/")
    if len(parts) > 1:
        return "lake/" + parts[1]
    return os.path.basename(image_path)
