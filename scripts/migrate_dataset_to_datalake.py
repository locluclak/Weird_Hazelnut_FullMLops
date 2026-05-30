import argparse
import mimetypes
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def migrate_dataset(
    dataset_root: Path,
    dataset_name: str,
    config_path: str | None = None,
) -> int:
    config = load_config(config_path)
    data_layer = create_data_layer(config)
    if not data_layer:
        raise RuntimeError("data_layer.enabled must be true before migrating a dataset.")

    migrated = 0
    for split in ("train", "test", "val"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        for label_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            label = label_dir.name
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                content = image_path.read_bytes()
                with Image.open(image_path) as img:
                    image = img.convert("RGB")
                    data_layer.persist_dataset_item(
                        dataset_name=dataset_name,
                        split=split,
                        label=label,
                        image=image,
                        image_bytes=content,
                        original_filename=image_path.name,
                        content_type=mimetypes.guess_type(image_path.name)[0] or "image/png",
                        source_path=str(image_path),
                    )
                migrated += 1

    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed MinIO/Postgres from the local /data dataset folder."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/hazelnut",
        help="Local seed dataset root. This is read once and remains for manual tests only.",
    )
    parser.add_argument(
        "--dataset-name",
        default="initial_hazelnut",
        help="Dataset version name recorded in Postgres.",
    )
    parser.add_argument("--config", default=None, help="Config path. Defaults to CONFIG_PATH/config.yaml.")
    args = parser.parse_args()

    count = migrate_dataset(
        dataset_root=Path(args.dataset_root),
        dataset_name=args.dataset_name,
        config_path=args.config,
    )
    print(f"Migrated {count} images into MinIO/Postgres dataset '{args.dataset_name}'.")


if __name__ == "__main__":
    main()
