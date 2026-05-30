from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset

from weird_hazelnut.data.layer import DataLayer


@dataclass(frozen=True)
class DatasetSample:
    split: str
    label: str
    object_key: str
    image_id: str


def load_samples(
    data_layer: DataLayer,
    split: str | None = None,
    labels: list[str] | None = None,
) -> list[DatasetSample]:
    rows = data_layer.list_dataset_images(split=split, labels=labels)
    return [
        DatasetSample(
            split=item.split,
            label=item.label,
            object_key=image.minio_object_key,
            image_id=image.id,
        )
        for item, image in rows
    ]


class MinioImageDataset(Dataset):
    def __init__(
        self,
        data_layer: DataLayer,
        samples: list[DatasetSample],
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ):
        self.data_layer = data_layer
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())
        self.transform = transform
        self.targets = [class_to_idx[sample.label] for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self.data_layer.open_image(sample.object_key).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[sample.label]


def dataset_counts(samples: Iterable[DatasetSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = f"dataset_count_{sample.split}_{sample.label}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def detect_leakage(samples: Iterable[DatasetSample]) -> list[tuple[DatasetSample, DatasetSample]]:
    train_by_image_id: dict[str, DatasetSample] = {}
    duplicates: list[tuple[DatasetSample, DatasetSample]] = []
    for sample in samples:
        if sample.split == "train":
            train_by_image_id[sample.image_id] = sample
        elif sample.image_id in train_by_image_id:
            duplicates.append((train_by_image_id[sample.image_id], sample))
    return duplicates


class AnomalibFolderStaging:
    """Ephemeral bridge for Anomalib Folder, which currently requires file paths."""

    def __init__(self, data_layer: DataLayer, dataset_name: str = "hazelnut"):
        self.data_layer = data_layer
        self.dataset_name = dataset_name
        self._tmp: TemporaryDirectory | None = None
        self.root: Path | None = None

    def __enter__(self) -> Path:
        self._tmp = TemporaryDirectory(prefix="weird_hazelnut_anomalib_")
        self.root = Path(self._tmp.name) / self.dataset_name
        self._write_split("train", ["good"])
        self._write_split("test", None)
        return self.root

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._tmp:
            self._tmp.cleanup()

    def _write_split(self, split: str, labels: list[str] | None) -> None:
        assert self.root is not None
        samples = load_samples(self.data_layer, split=split, labels=labels)
        for index, sample in enumerate(samples):
            target = self.root / split / sample.label / f"{index:08d}_{sample.image_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            image = self.data_layer.open_image(sample.object_key)
            image.save(target)
