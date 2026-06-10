import logging
import json
import os
import shutil
import hashlib

from weird_hazelnut.config import load_config
from weird_hazelnut.data.layer import create_data_layer
from weird_hazelnut.data.repositories import DataRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSyncWorker:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        ls_cfg = self.config.get("label_studio", {})

        self.url = ls_cfg.get("url")
        self.api_key = ls_cfg.get("api_key")
        self.project_id = ls_cfg.get("project_id")

        if self.api_key == "PASTE_YOUR_API_KEY_HERE":
            logger.error("Label Studio API key not set. Sync worker cannot run.")
            self.client = None
        else:
            try:
                from label_studio_sdk.client import LabelStudio

                self.client = LabelStudio(base_url=self.url, api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Label Studio client: %s", e)
                self.client = None

        self.output_base_dir = "data/final"
        self.class_map = {
            "Normal": "good",
            "crack": "crack",
            "cut": "cut",
            "hole": "hole",
            "print": "print",
        }

        data_cfg = self.config.get("data_sync", {})
        self.human_feedback_dataset_name = data_cfg.get(
            "human_feedback_dataset_name",
            "human_feedback_dataset",
        )
        self.human_feedback_split = data_cfg.get("human_feedback_split", "train")
        self.promote_annotations_to_dataset = bool(
            data_cfg.get("promote_annotations_to_dataset", True)
        )

        self.data_layer = create_data_layer(self.config)

    def sync(self):
        if not self.client:
            logger.error("Client not initialized. Aborting sync.")
            return

        try:
            logger.info("Fetching tasks for project %s...", self.project_id)
            tasks = self.client.tasks.list(project=self.project_id)

            synced_count = 0
            for task in tasks:
                if not task.annotations:
                    continue

                annotation = task.annotations[0]
                annotation_result = (
                    annotation.get("result")
                    if isinstance(annotation, dict)
                    else annotation.result
                )
                label = self._extract_label(annotation_result)


                if not label:
                    logger.warning("Could not extract label for task %s", task.id)
                    continue

                if self.data_layer:
                    if self._sync_annotation_to_database(task, label):
                        synced_count += 1
                    continue

                local_path = self._resolve_local_path(task.data.get("image", ""))
                if not os.path.exists(local_path):
                    logger.error("Source image not found: %s", local_path)
                    continue

                target_class = self.class_map.get(label, label.lower())
                target_dir = os.path.join(self.output_base_dir, "train", target_class)
                os.makedirs(target_dir, exist_ok=True)

                target_path = os.path.join(target_dir, os.path.basename(local_path))
                shutil.copy2(local_path, target_path)
                logger.info("Synced task %s: %s -> %s", task.id, local_path, target_path)
                synced_count += 1

            logger.info("Sync complete. %s tasks processed.", synced_count)
        except Exception as e:
            logger.error("Error during sync: %s", e)

    def _resolve_local_path(self, image_source_path: str) -> str:
        if "/static/" in image_source_path:
            rel_to_files = image_source_path.split("/static/")[1]
            return os.path.abspath(os.path.join("data/lake", rel_to_files))
        if "ls-local-files://" in image_source_path:
            rel_to_files = image_source_path.replace("ls-local-files://", "")
            if rel_to_files.startswith("lake/"):
                rel_path_in_lake = rel_to_files.replace("lake/", "", 1)
                return os.path.abspath(os.path.join("data/lake", rel_path_in_lake))
            return rel_to_files
        if "?d=" in image_source_path:
            rel_to_files = image_source_path.split("?d=")[1]
            if rel_to_files.startswith("lake/"):
                rel_path_in_lake = rel_to_files.replace("lake/", "", 1)
                return os.path.abspath(os.path.join("data/lake", rel_path_in_lake))
            return rel_to_files
        return image_source_path

    def _extract_label(self, result):
        if not result or not isinstance(result, list):
            return None
        
        primary_choice = None
        secondary_choice = None

        for item in result:
            if item.get("from_name") == "sentiment":
                choices = item.get("value", {}).get("choices", [])
                if choices:
                    primary_choice = choices[0]
            elif item.get("from_name") == "label":
                choices = item.get("value", {}).get("choices", [])
                if choices:
                    secondary_choice = choices[0]

        if primary_choice == "Anomaly" and secondary_choice:
            return secondary_choice
        return primary_choice

    def _sync_annotation_to_database(self, task, label: str) -> bool:
        target_class = self.class_map.get(label, label.lower())

        annotation = task.annotations[0]
        if isinstance(annotation, dict):
            annotation_id = annotation.get("id")
            annotator_id = annotation.get("completed_by")
            raw_annotation = annotation
        else:
            annotation_id = getattr(annotation, "id", None)
            annotator_id = getattr(annotation, "completed_by", None)
            raw_annotation = self._to_plain_dict(annotation)

        image_id = task.data.get("image_id") if getattr(task, "data", None) else None

        with self.data_layer.database.session() as session:
            repo = DataRepository(session)
            label_task = repo.get_label_task_by_studio_id(task.id)
            if not image_id and label_task:
                image_id = label_task.image_id
            if not image_id:
                logger.warning("Task %s has no image_id. Skipping database sync.", task.id)
                return False

            saved = repo.create_annotation(
                {
                    "image_id": image_id,
                    "label_task_id": label_task.id if label_task else None,
                    "source": "label_studio",
                    "quality_label": target_class,
                    "is_anomaly": target_class != "good",
                    "annotator_id": str(annotator_id) if annotator_id is not None else None,
                    "label_studio_annotation_id": annotation_id,
                    "raw_annotation": raw_annotation,
                }
            )

            dataset_item = None
            if self.promote_annotations_to_dataset:
                dataset = repo.get_or_create_dataset_version(
                    name=self.human_feedback_dataset_name,
                    description="Dataset built from Label Studio human annotations.",
                    split_strategy=f"human_feedback_{self.human_feedback_split}",
                    created_by="sync_worker.py",
                )

                feedback_split = self._assign_feedback_split(image_id, target_class)
                dataset_item = repo.add_dataset_item(
                    {
                        "dataset_version_id": dataset.id,
                        "image_id": image_id,
                        "annotation_id": saved.id,
                        "split": feedback_split,
                        "label": target_class,
                        "exported_path": None,
                    }
                )

            if label_task:
                repo.mark_label_task_synced(label_task.id)

        if dataset_item:
            logger.info(
                "Synced Label Studio task %s to annotation %s and dataset item %s/%s.",
                task.id,
                saved.id,
                dataset_item.split,
                dataset_item.label,
            )
        else:
            logger.info("Synced Label Studio task %s to annotation %s.", task.id, saved.id)

        return True
    
    def _assign_feedback_split(self, image_id: str, label: str) -> str:
        split_cfg = self.config.get("data_sync", {}).get(
            "human_feedback_split_strategy", {}
        )

        if not split_cfg.get("enabled", False):
            return self.human_feedback_split

        train_ratio = float(split_cfg.get("train_ratio", 0.8))
        val_ratio = float(split_cfg.get("val_ratio", 0.2))

        if train_ratio <= 0 or val_ratio < 0:
            raise ValueError("Invalid human feedback split ratio.")

        total_ratio = train_ratio + val_ratio
        val_threshold = int((val_ratio / total_ratio) * 10_000)

        key = f"{label}:{image_id}"
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10_000

        if bucket < val_threshold:
            return "val"
        return "train"

    def _to_plain_dict(self, value):
        if hasattr(value, "model_dump"):
            plain = value.model_dump(mode="json")
            return json.loads(json.dumps(plain, default=str))
        if hasattr(value, "dict"):
            plain = value.dict()
            return json.loads(json.dumps(plain, default=str))
        if isinstance(value, dict):
            return json.loads(json.dumps(value, default=str))
        plain = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        return json.loads(json.dumps(plain, default=str))
