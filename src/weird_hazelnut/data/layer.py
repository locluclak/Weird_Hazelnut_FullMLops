import logging
from pathlib import Path

from PIL import Image

from weird_hazelnut.data.database import Database
from weird_hazelnut.data.repositories import DataRepository
from weird_hazelnut.data.storage import (
    MinioObjectStore,
    image_to_png_bytes,
    object_key_for_hash,
    sha256_bytes,
)

logger = logging.getLogger(__name__)


class DataLayer:
    def __init__(self, database: Database, object_store: MinioObjectStore):
        self.database = database
        self.object_store = object_store

    def bootstrap(self) -> None:
        self.database.bootstrap()
        self.object_store.bootstrap()

    def persist_image(
        self,
        image: Image.Image,
        image_bytes: bytes | None = None,
        original_filename: str | None = None,
        content_type: str | None = None,
        storage_stage: str = "raw",
        object_key_prefix: str | None = None,
    ):
        content = image_bytes or image_to_png_bytes(image)
        sha256 = sha256_bytes(content)
        suffix = Path(original_filename or "upload.png").suffix or ".png"
        object_key = object_key_for_hash(sha256, suffix=suffix, prefix=object_key_prefix)
        mime_type = content_type or "image/png"

        self.object_store.upload_bytes(object_key, content, mime_type)
        with self.database.session() as session:
            repo = DataRepository(session)
            return repo.get_or_create_image(
                {
                    "sha256": sha256,
                    "original_filename": original_filename,
                    "content_type": mime_type,
                    "width": image.width,
                    "height": image.height,
                    "size_bytes": len(content),
                    "minio_bucket": self.object_store.bucket,
                    "minio_object_key": object_key,
                    "storage_stage": storage_stage,
                }
            )

    def update_image_stage(self, image_id: str, storage_stage: str) -> None:
        with self.database.session() as session:
            DataRepository(session).update_image_stage(image_id, storage_stage)

    def record_inference(self, values: dict):
        with self.database.session() as session:
            return DataRepository(session).create_inference_run(values)

    def record_label_task(
        self,
        image_id: str,
        label_studio_task_id: int,
        label_studio_project_id: int | None,
    ):
        with self.database.session() as session:
            return DataRepository(session).create_label_task(
                image_id=image_id,
                label_studio_task_id=label_studio_task_id,
                label_studio_project_id=label_studio_project_id,
            )

    def presigned_image_url(self, object_key: str, expires_seconds: int = 7 * 24 * 3600) -> str:
        return self.object_store.presigned_get_url(object_key, expires_seconds)

    def open_image(self, object_key: str) -> Image.Image:
        return self.object_store.open_image(object_key)

    def persist_dataset_item(
        self,
        dataset_name: str,
        split: str,
        label: str,
        image: Image.Image,
        image_bytes: bytes,
        original_filename: str | None = None,
        content_type: str | None = None,
        source_path: str | None = None,
        created_by: str = "migrate_dataset_to_datalake.py",
    ):
        image_record = self.persist_image(
            image=image,
            image_bytes=image_bytes,
            original_filename=original_filename,
            content_type=content_type,
            storage_stage=f"dataset/{split}",
            object_key_prefix=f"datasets/{dataset_name}/{split}/{label}",
        )
        with self.database.session() as session:
            repo = DataRepository(session)
            dataset = repo.get_or_create_dataset_version(
                name=dataset_name,
                description="Initial dataset imported from the local /data seed folder.",
                split_strategy="folder_import",
                created_by=created_by,
            )
            annotation = repo.get_or_create_annotation(
                {
                    "image_id": image_record.id,
                    "label_task_id": None,
                    "source": f"dataset_import:{dataset_name}",
                    "quality_label": label,
                    "is_anomaly": label != "good",
                    "annotator_id": None,
                    "label_studio_annotation_id": None,
                    "raw_annotation": {"split": split, "source_path": source_path},
                }
            )
            return repo.add_dataset_item(
                {
                    "dataset_version_id": dataset.id,
                    "image_id": image_record.id,
                    "annotation_id": annotation.id,
                    "split": split,
                    "label": label,
                    "exported_path": None,
                }
            )

    def list_dataset_images(
        self,
        split: str | None = None,
        labels: list[str] | None = None,
    ):
        with self.database.session() as session:
            return DataRepository(session).list_dataset_items(split=split, labels=labels)


def create_data_layer(config: dict) -> DataLayer | None:
    data_cfg = config.get("data_layer", {})
    if not data_cfg.get("enabled", False):
        return None

    database_url = data_cfg.get("database_url") or config.get("database", {}).get("url")
    storage_cfg = data_cfg.get("object_storage", {})
    if not database_url:
        raise ValueError("data_layer.database_url is required when data_layer.enabled is true")

    required_storage = ["endpoint", "access_key", "secret_key", "bucket"]
    missing = [key for key in required_storage if not storage_cfg.get(key)]
    if missing:
        raise ValueError(f"Missing object storage settings: {', '.join(missing)}")

    database = Database(database_url, echo=bool(data_cfg.get("echo_sql", False)))
    object_store = MinioObjectStore(
        endpoint=storage_cfg["endpoint"],
        access_key=storage_cfg["access_key"],
        secret_key=storage_cfg["secret_key"],
        bucket=storage_cfg["bucket"],
        secure=bool(storage_cfg.get("secure", False)),
        region=storage_cfg.get("region", "us-east-1"),
        public_endpoint=storage_cfg.get("public_endpoint"),
        public_secure=storage_cfg.get("public_secure"),
    )
    layer = DataLayer(database, object_store)
    layer.bootstrap()
    logger.info("Data layer initialized with Postgres and MinIO.")
    return layer
