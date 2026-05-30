# src/data_utils.py
import hashlib
import pathlib
from typing import Dict, Iterable, List, Tuple

def iter_images(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Iterate over all images in the root directory and subdirectories."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path

def hash_file(path: pathlib.Path) -> str:
    """Calculate MD5 hash of a file to check for duplicates."""
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def detect_data_leakage(data_root_str: str) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """
    Check if any images in the train set are also in the test/val sets.
    Returns:
         List of tuples: [(train_image_path, duplicate_image_path)]
    """
    data_root = pathlib.Path(data_root_str).resolve()
    train_root = data_root / "train"
    val_root = data_root / "val"
    test_root = data_root / "test"

    if not train_root.exists():
        return []

    # Map train hashes to image paths
    train_hashes: Dict[str, pathlib.Path] = {}
    for img_path in iter_images(train_root):
        train_hashes[hash_file(img_path)] = img_path

    duplicates: List[Tuple[pathlib.Path, pathlib.Path]] = []
    
    # Check val hashes against train hashes
    if val_root.exists():
        for img_path in iter_images(val_root):
            digest = hash_file(img_path)
            if digest in train_hashes:
                duplicates.append((train_hashes[digest], img_path))
                
    # Check test hashes against train hashes
    if test_root.exists():
        for img_path in iter_images(test_root):
            digest = hash_file(img_path)
            if digest in train_hashes:
                duplicates.append((train_hashes[digest], img_path))

    return duplicates
