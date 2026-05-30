from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

from weird_hazelnut.data.layer import DataLayer
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

    all_samples = load_samples(data_layer)
    labels = sorted({sample.label for sample in all_samples if include_good or sample.label != "good"})
    train_samples = [s for s in all_samples if s.split == "train" and s.label in labels]
    val_split = "val" if any(s.split == "val" and s.label in labels for s in all_samples) else "test"
    val_samples = [s for s in all_samples if s.split == val_split and s.label in labels]
    if not train_samples or not val_samples:
        logger.error("Classifier training needs non-empty train and validation/test samples.")
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
        onnx_path = output_dir / training_cfg.get("onnx_name", "model.onnx")
        meta_path = output_dir / training_cfg.get("meta_name", "meta.json")
        export_onnx(model, img_size, onnx_path)
        meta = {"classes": labels, "img_size": img_size, "val_acc": best_acc, "history": history, "mlflow_run_id": run.info.run_id}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mlflow.log_metric("eval_best_val_acc", float(best_acc))
        mlflow.log_artifact(str(onnx_path), "production_models")
        mlflow.log_artifact(str(meta_path), "production_models")

    logger.info("Classifier retraining complete. Artifacts written to %s", output_dir)
    return True
