import argparse
import shutil
import random
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

def setup_mvtec(src_dir: Path, dst_dir: Path):
    """
    Convert MVTec AD dataset -> ImageFolder
    MVTec structure:
    hazelnut/
        train/
            good/
        test/
            good/
            crack/
            ...
        ground_truth/
            ...
    
    Target structure:
    dst_dir/
        train/
            good/
            crack/
            ...
        test/
            good/
            crack/
            ...
    """
    print(f"Converting MVTec from {src_dir} to {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train/good
    train_good_src = src_dir / "train" / "good"
    train_good_dst = dst_dir / "train" / "good"
    if train_good_src.exists():
        train_good_dst.mkdir(parents=True, exist_ok=True)
        for img_path in train_good_src.glob("*.png"):
            shutil.copy(img_path, train_good_dst / img_path.name)

    # Process test (all classes)
    test_src = src_dir / "test"
    test_dst = dst_dir / "test"
    if test_src.exists():
        for class_dir in test_src.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                target_class_dir = test_dst / class_name
                target_class_dir.mkdir(parents=True, exist_ok=True)
                for img_path in class_dir.glob("*.png"):
                    shutil.copy(img_path, target_class_dir / img_path.name)
    
    print("MVTec conversion complete.")

def split_dataset(src_dir: Path, dst_dir: Path, val_split: float, seed: int):
    """
    Auto train/val split
    """
    random.seed(seed)
    print(f"Splitting dataset from {src_dir} to {dst_dir} (val_split={val_split})")
    
    for split in ["train", "test"]:
        split_src = src_dir / split
        if not split_src.exists():
            continue
            
        for class_dir in split_src.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            random.shuffle(images)
            
            n_val = int(len(images) * val_split)
            val_images = images[:n_val]
            train_images = images[n_val:]
            
            # For 'test' split, we usually just copy it as is to 'test'
            # But the requirement says "Auto train/val split". 
            # Usually this means splitting the 'train' folder into 'train' and 'val'.
            
            if split == "train":
                train_dst = dst_dir / "train" / class_name
                val_dst = dst_dir / "val" / class_name
                train_dst.mkdir(parents=True, exist_ok=True)
                val_dst.mkdir(parents=True, exist_ok=True)
                
                for img in train_images:
                    shutil.copy(img, train_dst / img.name)
                for img in val_images:
                    shutil.copy(img, val_dst / img.name)
            else:
                # Just copy test as is
                test_dst = dst_dir / "test" / class_name
                test_dst.mkdir(parents=True, exist_ok=True)
                for img in images:
                    shutil.copy(img, test_dst / img.name)

    print("Dataset split complete.")

def import_csv(csv_path: Path, dst_dir: Path, symlink: bool):
    """
    Import from CSV manifest (columns: path, label)
    """
    print(f"Importing from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_src = Path(row['path'])
        label = row['label']
        
        # Assume these go to a generic 'data' folder or we need to know if it's train/test
        # Spec says "Import from CSV manifest... to dst_dir"
        target_dir = dst_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        
        img_dst = target_dir / img_src.name
        if symlink:
            img_dst.symlink_to(img_src.absolute())
        else:
            shutil.copy(img_src, img_dst)
    print("CSV import complete.")

def check_dataset(data_dir: Path):
    """
    Print dataset statistics + imbalance warnings
    """
    print(f"Checking dataset: {data_dir}")
    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\nSplit: {split}")
        split_stats = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                split_stats[class_dir.name] = count
                print(f"  {class_dir.name}: {count}")
        
        if split_stats:
            counts = list(split_stats.values())
            max_c = max(counts)
            min_c = min(counts)
            if min_c > 0 and max_c / min_c > 5:
                print(f"  WARNING: High imbalance in {split} (ratio {max_c/min_c:.2f} > 5)")
            if min_c < 50:
                print(f"  WARNING: Too few samples in some classes of {split} (min {min_c} < 50)")
        stats[split] = split_stats

def full_split(src_dir: Path, dst_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """
    Split a single folder with class subdirectories into train/val/test
    """
    random.seed(seed)
    print(f"Full split from {src_dir} to {dst_dir}")
    print(f"Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }
        
        for split_name, split_images in splits.items():
            target_dir = dst_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy(img, target_dir / img.name)
                
    print("Full split complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hazelnut Data Preparation")
    subparsers = parser.add_subparsers(dest="command")
    
    # mvtec
    p_mvtec = subparsers.add_parser("mvtec")
    p_mvtec.add_argument("src", type=Path)
    p_mvtec.add_argument("dst", type=Path)
    
    # split
    p_split = subparsers.add_parser("split")
    p_split.add_argument("src", type=Path)
    p_split.add_argument("dst", type=Path)
    p_split.add_argument("--val", type=float, default=0.2)
    p_split.add_argument("--seed", type=int, default=42)

    # full_split
    p_full = subparsers.add_parser("full_split")
    p_full.add_argument("src", type=Path)
    p_full.add_argument("dst", type=Path)
    p_full.add_argument("--train", type=float, default=0.7)
    p_full.add_argument("--val", type=float, default=0.1)
    p_full.add_argument("--test", type=float, default=0.2)
    p_full.add_argument("--seed", type=int, default=42)
    
    # csv
    p_csv = subparsers.add_parser("csv")
    p_csv.add_argument("csv", type=Path)
    p_csv.add_argument("dst", type=Path)
    p_csv.add_argument("--symlink", action="store_true")
    
    # check
    p_check = subparsers.add_parser("check")
    p_check.add_argument("data", type=Path)
    
    args = parser.parse_args()
    
    if args.command == "mvtec":
        setup_mvtec(args.src, args.dst)
    elif args.command == "split":
        split_dataset(args.src, args.dst, args.val, args.seed)
    elif args.command == "full_split":
        full_split(args.src, args.dst, args.train, args.val, args.test, args.seed)
    elif args.command == "csv":
        import_csv(args.csv, args.dst, args.symlink)
    elif args.command == "check":
        check_dataset(args.data)
    else:
        parser.print_help()
