import logging
from datetime import datetime
from pathlib import Path

from weird_hazelnut.data.layer import DataLayer
from weird_hazelnut.data.repositories import DataRepository

logger = logging.getLogger(__name__)


class DatasetExporter:
    def __init__(
        self,
        data_layer: DataLayer,
        output_dir: str = "data/final",
        allow_local_export: bool = False,
    ):
        self.data_layer = data_layer
        self.output_dir = Path(output_dir)
        self.allow_local_export = allow_local_export

    def export_training_dataset(
        self,
        name: str | None = None,
        description: str | None = None,
        created_by: str = "retrain.py",
    ) -> Path:
        if not self.allow_local_export:
            raise RuntimeError(
                "Local dataset export is disabled. Use MinIO/Postgres dataset readers "
                "instead of downloading data into data/final."
            )

        dataset_name = name or f"training_export_{datetime.utcnow():%Y%m%d_%H%M%S}"
        with self.data_layer.database.session() as session:
            repo = DataRepository(session)
            dataset = repo.create_dataset_version(
                name=dataset_name,
                description=description,
                split_strategy="all_train",
                created_by=created_by,
            )
            rows = repo.list_training_annotations()

            for annotation, image in rows:
                target = (
                    self.output_dir
                    / "train"
                    / annotation.quality_label
                    / f"{image.sha256}{Path(image.minio_object_key).suffix or '.png'}"
                )
                self.data_layer.object_store.download_to_path(
                    image.minio_object_key,
                    target,
                )
                repo.add_dataset_item(
                    {
                        "dataset_version_id": dataset.id,
                        "image_id": image.id,
                        "annotation_id": annotation.id,
                        "split": "train",
                        "label": annotation.quality_label,
                        "exported_path": str(target),
                    }
                )

        logger.info("Exported %s labeled samples into %s", len(rows), self.output_dir)
        return self.output_dir
