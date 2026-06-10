from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from weird_hazelnut.data.layer import DataLayer
from weird_hazelnut.data.repositories import DataRepository
from weird_hazelnut.training.minio_dataset import (
    MinioImageDataset,
    dataset_counts,
    detect_leakage,
    load_samples,
)

logger = logging.getLogger(__name__)


def optimize_cpu() -> None:
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)
    try:
        torch.set_num_interop_threads(max(1, cpu_count // 2))
    except RuntimeError:
        pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(img_size: int, level: str = "medium"):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [transforms.Resize((img_size, img_size))]
    if level == "heavy":
        train_ops = base + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.2),
        ]
    elif level == "light":
        train_ops = base + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            norm,
        ]
    else:
        train_ops = base + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            norm,
        ]
    val_ops = base + [transforms.ToTensor(), norm]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def get_model(num_classes: int, pretrained: bool = False):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
    return model


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    if scheduler:
        scheduler.step()
    return running_loss / max(total, 1), correct / max(total, 1)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def export_onnx(model, img_size: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model.to("cpu"),
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )


def prepare_local_classifier_dataset(config: dict, data_layer: DataLayer) -> bool:
    """
    Compiles a local training/dev dataset under dataset_dir:
    1. Copies & splits (80/20) seed test anomaly images (crack, cut, hole, print) from data/hazelnut/test/.
    2. Downloads active-learning human annotations (where is_anomaly == True).
       - Uses split priority: If a class has < train_priority_threshold synced annotations, 100% go to train/.
       - Otherwise, split 80/20.
    """
    training_cfg = config.get("training", {}).get("classifier", {})
    dataset_dir = training_cfg.get("dataset_dir", "data/classifier_dataset")
    split_ratio = float(training_cfg.get("split_ratio", 0.8))
    train_priority_threshold = int(training_cfg.get("train_priority_threshold", 10))

    dataset_path = Path(dataset_dir)
    labels = ["crack", "cut", "hole", "print"]

    # 1. Clean and recreate the target directories
    try:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        for split in ["train", "val"]:
            for label in labels:
                (dataset_path / split / label).mkdir(parents=True, exist_ok=True)
        logger.info(f"Re-initialized classifier dataset folders under {dataset_path}")
    except Exception as e:
        logger.error(f"Error re-initializing dataset directories: {e}")
        return False

    # 2. Extract and split seed test images
    seed_test_dir = Path("data/hazelnut/test")
    if not seed_test_dir.exists():
        logger.error(f"Seed test directory {seed_test_dir} does not exist.")
        return False

    for label in labels:
        label_dir = seed_test_dir / label
        if not label_dir.exists():
            logger.warning(f"Seed label directory {label_dir} does not exist. Skipping.")
            continue
        
        # Get all image files
        img_files = [f for f in label_dir.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        # Shuffle seed files to ensure a random 80/20 split
        random.shuffle(img_files)
        
        train_count = int(len(img_files) * split_ratio)
        train_files = img_files[:train_count]
        val_files = img_files[train_count:]

        # Copy to train/val
        for f in train_files:
            shutil.copy2(f, dataset_path / "train" / label / f.name)
        for f in val_files:
            shutil.copy2(f, dataset_path / "val" / label / f.name)
        
        logger.info(f"Seeded class '{label}': {len(train_files)} to train, {len(val_files)} to val.")

    # 3. Fetch active annotations and download corresponding images
    try:
        with data_layer.database.session() as session:
            repo = DataRepository(session)
            annotations = repo.list_training_annotations()
    except Exception as e:
        logger.error(f"Error fetching active annotations from database: {e}")
        return False

    active_annotations_by_label = {label: [] for label in labels}
    for annotation, image_record in annotations:
        if annotation.source == "label_studio" and annotation.is_anomaly and annotation.quality_label in active_annotations_by_label:
            active_annotations_by_label[annotation.quality_label].append((annotation, image_record))

    for label, items in active_annotations_by_label.items():
        if not items:
            continue
        
        num_items = len(items)
        random.shuffle(items)

        # Decide split allocation
        if num_items < train_priority_threshold:
            train_items = items
            val_items = []
            logger.info(f"Class '{label}' has {num_items} annotations (< {train_priority_threshold}). Allocating 100% to train.")
        else:
            train_count = int(num_items * split_ratio)
            train_items = items[:train_count]
            val_items = items[train_count:]
            logger.info(f"Class '{label}' has {num_items} annotations. Splitting: {len(train_items)} to train, {len(val_items)} to val.")

        # Download images
        for annotation, image_record in train_items:
            try:
                suffix = Path(image_record.original_filename or "image.png").suffix or ".png"
                target_filename = f"{image_record.id}{suffix}"
                target_path = dataset_path / "train" / label / target_filename
                data_layer.object_store.download_to_path(image_record.minio_object_key, target_path)
            except Exception as e:
                logger.warning(f"Failed to download image {image_record.id} for annotation: {e}")

        for annotation, image_record in val_items:
            try:
                suffix = Path(image_record.original_filename or "image.png").suffix or ".png"
                target_filename = f"{image_record.id}{suffix}"
                target_path = dataset_path / "val" / label / target_filename
                data_layer.object_store.download_to_path(image_record.minio_object_key, target_path)
            except Exception as e:
                logger.warning(f"Failed to download image {image_record.id} for annotation: {e}")

    logger.info("Successfully completed preparing local classifier dataset.")
    return True


def train_classifier(config: dict, data_layer: DataLayer) -> bool:
    optimize_cpu()
    set_seed(42)

    training_cfg = config.get("training", {}).get("classifier", {})
    mlflow_cfg = config.get("mlflow", {})
    output_dir = Path(training_cfg.get("output_dir", "models/classifier_retrained"))
    img_size = int(training_cfg.get("img_size", 224))
    epochs = int(training_cfg.get("epochs", 5))
    batch_size = int(training_cfg.get("batch_size", 32))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    label_smooth = float(training_cfg.get("label_smooth", 0.1))
    early_stop = int(training_cfg.get("early_stop", 5))
    augment_level = training_cfg.get("augment_level", "medium")
    include_good = bool(training_cfg.get("include_good", False))
    pretrained = bool(training_cfg.get("pretrained", False))

    split_ratio = float(training_cfg.get("split_ratio", 0.8))

    all_samples = load_samples(data_layer)
    labels = sorted({sample.label for sample in all_samples if include_good or sample.label != "good"})
    
    # 1. Filter human annotations from Label Studio (which are in train/val splits)
    human_train = [s for s in all_samples if s.split == "train" and s.label in labels]
    human_val = [s for s in all_samples if s.split == "val" and s.label in labels]
    
    # 2. Filter seed test anomalies (which have split == "test")
    seed_anomalies = [s for s in all_samples if s.split == "test" and s.label in labels]
    
    # Split seed anomalies dynamically using a deterministic shuffle with seed 42
    import random
    rng = random.Random(42)
    
    seed_train = []
    seed_val = []
    
    # Group by label to ensure a perfectly stratified/balanced split per class
    anomalies_by_label = {}
    for s in seed_anomalies:
        anomalies_by_label.setdefault(s.label, []).append(s)
        
    for label, samples_list in sorted(anomalies_by_label.items()):
        # Sort first to ensure deterministic shuffle regardless of DB order
        samples_list = sorted(samples_list, key=lambda x: x.object_key)
        rng.shuffle(samples_list)
        
        train_count = int(len(samples_list) * split_ratio)
        seed_train.extend(samples_list[:train_count])
        seed_val.extend(samples_list[train_count:])
        
    # Combine seed splits with human feedback splits
    train_samples = seed_train + human_train
    val_samples = seed_val + human_val

    if not train_samples or not val_samples:
        logger.error("Classifier training needs non-empty train and validation samples.")
        return False

    class_to_idx = {name: idx for idx, name in enumerate(labels)}
    train_tsfm, val_tsfm = get_transforms(img_size, augment_level)
    train_dataset = MinioImageDataset(data_layer, train_samples, class_to_idx, train_tsfm)
    val_dataset = MinioImageDataset(data_layer, val_samples, class_to_idx, val_tsfm)

    class_counts = np.array(
        [sum(1 for target in train_dataset.targets if target == idx) for idx in range(len(labels))],
        dtype=np.float32,
    )
    if np.any(class_counts == 0):
        logger.error("Classifier train split is missing one or more classes: %s", labels)
        return False

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[target] for target in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = get_model(len(labels), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=label_smooth,
    )

    def set_freeze(freeze: bool) -> None:
        for param in model.parameters():
            param.requires_grad = not freeze
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.features[13:].parameters():
            param.requires_grad = True

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "WeirdHazelnut_Integrated_Pipeline"))
    run_name = training_cfg.get("run_name", "Retraining_Classifier_MinIO")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    with mlflow.start_run(run_name=run_name) as run:
        # Log system specifications and environment specs
        try:
            import platform
            import psutil
            mlflow.log_params({
                "system_processor": platform.processor() or "unknown",
                "system_platform": platform.platform(),
                "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "python_version": platform.python_version()
            })
        except Exception as sys_err:
            logger.warning(f"Could not log system specifications: {sys_err}")

        # Log database dataset lineage metadata
        try:
            with data_layer.database.session() as session:
                repo = DataRepository(session)
                dataset = repo.get_or_create_dataset_version(name="initial_hazelnut")
                mlflow.log_params({
                    "dataset_version_name": dataset.name,
                    "dataset_version_id": dataset.id,
                    "dataset_created_at": str(dataset.created_at)
                })
        except Exception as db_err:
            logger.warning(f"Could not log database dataset version metadata: {db_err}")

        duplicates = detect_leakage(all_samples)
        mlflow.log_params(
            {
                "trainer": "integrated_classifier",
                "data_source": "minio_postgres",
                "classes": ",".join(labels),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "img_size": img_size,
                "augment_level": augment_level,
                "label_smooth": label_smooth,
                "early_stop": early_stop,
                "pretrained": pretrained,
                "data_leakage_detected": bool(duplicates),
                "data_leakage_count": len(duplicates),
            }
        )
        mlflow.log_params(dataset_counts(all_samples))

        set_freeze(True)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        unfreeze_epoch = max(1, epochs // 3)
        best_acc = -1.0
        patience_counter = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            if epoch == unfreeze_epoch:
                set_freeze(False)
                optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - unfreeze_epoch, 1))
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                },
                step=epoch,
            )
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stop:
                break

        model.load_state_dict(torch.load(best_path, map_location=device))
        
        # Collect predictions for MLflow evaluation plots
        model.eval()
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(1).cpu().numpy()
                y_true_list.extend(targets.numpy())
                y_pred_list.extend(preds)

        from weird_hazelnut.training.visualization import plot_confusion_matrix, plot_per_class_accuracy
        
        cm_path = output_dir / "confusion_matrix.png"
        acc_path = output_dir / "per_class_accuracy.png"
        
        plot_confusion_matrix(
            y_true=y_true_list,
            y_pred=y_pred_list,
            classes=labels,
            title="Classifier Confusion Matrix",
            output_path=cm_path
        )
        plot_per_class_accuracy(
            y_true=y_true_list,
            y_pred=y_pred_list,
            classes=labels,
            title="Classifier Per-Class Accuracy",
            output_path=acc_path
        )

        onnx_path = output_dir / training_cfg.get("onnx_name", "model.onnx")
        meta_path = output_dir / training_cfg.get("meta_name", "meta.json")
        export_onnx(model, img_size, onnx_path)
        
        # Automated Latency Benchmark for exported ONNX model
        try:
            logger.info("Benchmarking exported ONNX classifier model latency...")
            import onnxruntime as ort
            import time
            
            # Setup session options for single thread CPU benchmarking
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
            
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
            
            # Warmup
            for _ in range(5):
                session.run(None, {input_name: dummy_input})
                
            # Benchmark latency
            latencies = []
            for _ in range(30):
                t0 = time.perf_counter()
                session.run(None, {input_name: dummy_input})
                latencies.append((time.perf_counter() - t0) * 1000.0)
                
            avg_latency = float(np.mean(latencies))
            p95_latency = float(np.percentile(latencies, 95))
            mlflow.log_metric("cls_inference_latency_avg_ms", avg_latency)
            mlflow.log_metric("cls_inference_latency_p95_ms", p95_latency)
            logger.info(f"ONNX inference latency logged: Avg {avg_latency:.2f}ms, P95 {p95_latency:.2f}ms")
        except Exception as benchmark_err:
            logger.warning(f"Failed to benchmark ONNX classifier model latency: {benchmark_err}")

        meta = {"classes": labels, "img_size": img_size, "val_acc": best_acc, "history": history, "mlflow_run_id": run.info.run_id}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mlflow.log_metric("eval_best_val_acc", float(best_acc))
        mlflow.log_artifact(str(onnx_path), "production_models")
        onnx_data_path = onnx_path.with_name(onnx_path.name + ".data")
        if onnx_data_path.exists():
            mlflow.log_artifact(str(onnx_data_path), "production_models")
        mlflow.log_artifact(str(meta_path), "production_models")
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path), "production_models")
        if acc_path.exists():
            mlflow.log_artifact(str(acc_path), "production_models")

    logger.info("Classifier retraining complete. Artifacts written to %s", output_dir)
    return True
