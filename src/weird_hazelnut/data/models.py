import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class ImageRecord(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(100))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    minio_bucket: Mapped[str] = mapped_column(String(128), nullable=False)
    minio_object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    storage_stage: Mapped[str] = mapped_column(String(50), nullable=False, default="raw")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_predicted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    uncertain: Mapped[bool] = mapped_column(Boolean, nullable=False)
    routed_category: Mapped[str] = mapped_column(String(50), nullable=False)
    classifier_called: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    classifier_label: Mapped[str | None] = mapped_column(String(100))
    classifier_confidence: Mapped[float | None] = mapped_column(Float)
    ad_latency_ms: Mapped[float | None] = mapped_column(Float)
    classifier_latency_ms: Mapped[float | None] = mapped_column(Float)
    total_latency_ms: Mapped[float | None] = mapped_column(Float)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class LabelTask(Base):
    __tablename__ = "label_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), nullable=False)
    label_studio_task_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    label_studio_project_id: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now, onupdate=_now)


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), nullable=False)
    label_task_id: Mapped[str | None] = mapped_column(ForeignKey("label_tasks.id"))
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    quality_label: Mapped[str] = mapped_column(String(100), nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    annotator_id: Mapped[str | None] = mapped_column(String(100))
    label_studio_annotation_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    raw_annotation: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    split_strategy: Mapped[str] = mapped_column(String(100), nullable=False, default="all_train")
    created_by: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class DatasetItem(Base):
    __tablename__ = "dataset_items"

    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id"), primary_key=True
    )
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), primary_key=True)
    annotation_id: Mapped[str] = mapped_column(ForeignKey("annotations.id"), nullable=False)
    split: Mapped[str] = mapped_column(String(50), nullable=False)
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    exported_path: Mapped[str | None] = mapped_column(String(1024))


Index("idx_images_sha256", ImageRecord.sha256)
Index("idx_images_storage_stage", ImageRecord.storage_stage)
Index("idx_inference_runs_image_id", InferenceRun.image_id)
Index("idx_label_tasks_image_id", LabelTask.image_id)
Index("idx_annotations_image_id", Annotation.image_id)
Index("idx_annotations_quality_label", Annotation.quality_label)
Index("idx_dataset_items_version_split_label", DatasetItem.dataset_version_id, DatasetItem.split, DatasetItem.label)
