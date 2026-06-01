from sqlalchemy import select
from sqlalchemy.orm import Session

from weird_hazelnut.data.models import (
    Annotation,
    DatasetItem,
    DatasetVersion,
    ImageRecord,
    InferenceRun,
    LabelTask,
)


class DataRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create_image(self, values: dict) -> ImageRecord:
        existing = self.session.scalar(
            select(ImageRecord).where(ImageRecord.sha256 == values["sha256"])
        )
        if existing:
            existing.storage_stage = values.get("storage_stage", existing.storage_stage)
            return existing

        image = ImageRecord(**values)
        self.session.add(image)
        self.session.flush()
        return image

    def update_image_stage(self, image_id: str, storage_stage: str) -> None:
        image = self.session.get(ImageRecord, image_id)
        if image:
            image.storage_stage = storage_stage

    def create_inference_run(self, values: dict) -> InferenceRun:
        run = InferenceRun(**values)
        self.session.add(run)
        self.session.flush()
        return run

    def create_label_task(
        self,
        image_id: str,
        label_studio_task_id: int,
        label_studio_project_id: int | None,
        status: str = "created",
    ) -> LabelTask:
        existing = self.session.scalar(
            select(LabelTask).where(LabelTask.label_studio_task_id == label_studio_task_id)
        )
        if existing:
            existing.status = status
            return existing

        task = LabelTask(
            image_id=image_id,
            label_studio_task_id=label_studio_task_id,
            label_studio_project_id=label_studio_project_id,
            status=status,
        )
        self.session.add(task)
        self.session.flush()
        return task

    def get_label_task_by_studio_id(self, task_id: int) -> LabelTask | None:
        return self.session.scalar(
            select(LabelTask).where(LabelTask.label_studio_task_id == task_id)
        )

    def create_annotation(self, values: dict) -> Annotation:
        annotation_id = values.get("label_studio_annotation_id")
        if annotation_id is not None:
            existing = self.session.scalar(
                select(Annotation).where(
                    Annotation.label_studio_annotation_id == annotation_id
                )
            )
            if existing:
                return existing

        annotation = Annotation(**values)
        self.session.add(annotation)
        self.session.flush()
        return annotation

    def get_or_create_annotation(self, values: dict) -> Annotation:
        existing = self.session.scalar(
            select(Annotation).where(
                Annotation.image_id == values["image_id"],
                Annotation.source == values["source"],
                Annotation.quality_label == values["quality_label"],
            )
        )
        if existing:
            return existing
        return self.create_annotation(values)

    def mark_label_task_synced(self, label_task_id: str) -> None:
        task = self.session.get(LabelTask, label_task_id)
        if task:
            task.status = "synced"

    def create_dataset_version(
        self,
        name: str,
        description: str | None = None,
        split_strategy: str = "all_train",
        created_by: str | None = None,
    ) -> DatasetVersion:
        dataset = DatasetVersion(
            name=name,
            description=description,
            split_strategy=split_strategy,
            created_by=created_by,
        )
        self.session.add(dataset)
        self.session.flush()
        return dataset

    def get_or_create_dataset_version(
        self,
        name: str,
        description: str | None = None,
        split_strategy: str = "folder_import",
        created_by: str | None = None,
    ) -> DatasetVersion:
        existing = self.session.scalar(
            select(DatasetVersion).where(DatasetVersion.name == name)
        )
        if existing:
            return existing
        return self.create_dataset_version(
            name=name,
            description=description,
            split_strategy=split_strategy,
            created_by=created_by,
        )

    def list_training_annotations(self) -> list[tuple[Annotation, ImageRecord]]:
        stmt = (
            select(Annotation, ImageRecord)
            .join(ImageRecord, ImageRecord.id == Annotation.image_id)
            .order_by(Annotation.created_at.asc())
        )
        return list(self.session.execute(stmt).all())

    def add_dataset_item(self, values: dict) -> DatasetItem:
        item = DatasetItem(**values)
        self.session.merge(item)
        self.session.flush()
        return item

    def list_dataset_items(
        self,
        split: str | None = None,
        labels: list[str] | None = None,
    ) -> list[tuple[DatasetItem, ImageRecord]]:
        stmt = select(DatasetItem, ImageRecord).join(
            ImageRecord, ImageRecord.id == DatasetItem.image_id
        )
        if split:
            stmt = stmt.where(DatasetItem.split == split)
        if labels:
            stmt = stmt.where(DatasetItem.label.in_(labels))
        stmt = stmt.order_by(DatasetItem.label.asc(), ImageRecord.created_at.asc())
        return list(self.session.execute(stmt).all())
