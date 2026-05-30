import argparse
import json
import os
import random
import time
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Add current folder to sys.path for robust modular imports
sys.path.append(str(Path(__file__).parent))

import mlflow
from data_utils import detect_data_leakage, iter_images

# CPU Optimization
def optimize_cpu():
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(max(1, os.cpu_count() // 2))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_model(num_classes):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def get_transforms(img_size, level='medium'):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if level == 'light':
        train_tsfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            norm
        ])
    elif level == 'heavy':
        train_tsfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.2)
        ])
    else: # medium
        train_tsfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            norm
        ])
    
    val_tsfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm
    ])
    
    return train_tsfm, val_tsfm

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            
        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)
        
    if scheduler:
        scheduler.step()
        
    return running_loss / total, correct / total

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
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total

def export_onnx(model, img_size, output_path):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model.to('cpu'), dummy_input, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"Exported ONNX to {output_path}")

def train(args):
    optimize_cpu()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using training device target: {device}")
    
    # 1. Load centralized configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    print("--- Loaded Configuration Settings ---")
    print(yaml.dump(cfg, default_flow_style=False))
    
    # Extract config sections
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    mlflow_cfg = cfg.get("mlflow", {})
    export_cfg = cfg.get("export", {})
    
    # Overwrite config values with CLI values if they were explicitly provided
    data_dir = Path(args.data_dir if args.data_dir is not None else data_cfg.get("root", "data/final"))
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 1e-4)
    img_size = args.img_size if args.img_size is not None else data_cfg.get("img_size", 224)
    augment_level = args.augment_level if args.augment_level is not None else data_cfg.get("augment_level", "medium")
    label_smooth = args.label_smooth if args.label_smooth is not None else train_cfg.get("label_smooth", 0.1)
    mixup_alpha = args.mixup_alpha if args.mixup_alpha is not None else train_cfg.get("mixup_alpha", 0.2)
    early_stop = args.early_stop if args.early_stop is not None else train_cfg.get("early_stop", 5)
    fast_cpu = args.fast_cpu or data_cfg.get("fast_cpu", False)
    
    if fast_cpu:
        img_size = 160
        print(f"[FAST_CPU] Override active. Target image size resized to {img_size}x{img_size}")

    # 2. Start Data Leakage Verification
    print("\n--- Running Data Leakage Checker ---")
    duplicates = detect_data_leakage(str(data_dir))
    has_leakage = len(duplicates) > 0
    if has_leakage:
        print(f"[WARNING] Data leakage detected! Found {len(duplicates)} duplicate samples between Train and Val/Test splits.")
        for i, (train_img, other_img) in enumerate(duplicates[:5]):
            print(f"  Leakage {i}: train={train_img.name} <=> test/val={other_img.name}")
    else:
        print("[SUCCESS] No data leakage detected! Dataset splits are correctly partitioned.")

    # 3. Gather Dataset Statistics
    print("\n--- Gathering Dataset Statistics ---")
    split_counts = {}
    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if split_path.exists():
            print(f"Split '{split}':")
            for cls_dir in split_path.iterdir():
                if cls_dir.is_dir():
                    count = len(list(iter_images(cls_dir)))
                    split_counts[f"dataset_count_{split}_{cls_dir.name}"] = count
                    print(f"  {cls_dir.name}: {count}")

    # 4. Setup MLflow context with Resilient Connection Fallback
    tracking_uri = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")
    if tracking_uri.startswith("http"):
        import urllib.request
        try:
            # Quick 2-second timeout ping to verify server is active
            urllib.request.urlopen(tracking_uri, timeout=2.0)
            print(f"[MLflow] Connected successfully to tracking server at: {tracking_uri}")
        except Exception:
            print(f"[MLflow] [WARNING] Tracking server at {tracking_uri} is offline or unreachable.")
            print("[MLflow] Falling back to local file-based tracking at 'file:./mlruns' to prevent training crash.")
            tracking_uri = "file:./mlruns"
            
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "Hazelnut_Defect_Classification"))
    
    print("\n==============================================================================")
    print(f"Starting MLflow Run: {mlflow_cfg.get('run_name', 'MobileNetV3_Training_Run')}")
    print("==============================================================================")
    
    with mlflow.start_run(run_name=mlflow_cfg.get("run_name", "MobileNetV3_Training_Run")) as run:
        # A. Log All Hyperparameters as parameters
        mlflow.log_params({
            "model_name": model_cfg.get("name", "mobilenet_v3_large"),
            "data_dir": str(data_dir),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "img_size": img_size,
            "augment_level": augment_level,
            "label_smooth": label_smooth,
            "mixup_alpha": mixup_alpha,
            "early_stop": early_stop,
            "fast_cpu": fast_cpu,
            "data_leakage_detected": has_leakage,
            "data_leakage_count": len(duplicates)
        })
        
        # B. Log Dataset counts
        mlflow.log_params(split_counts)
        
        # C. Prepare DataLoaders
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        if not val_dir.exists():
            val_dir = data_dir / "test"
            
        train_tsfm, val_tsfm = get_transforms(img_size, augment_level)
        train_dataset = datasets.ImageFolder(train_dir, transform=train_tsfm)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_tsfm)
        
        # Ensure class index mapping matches between splits!
        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            print("[WARNING] Class index mappings between Train and Val datasets do not match!")
            print(f"  Train mapping: {train_dataset.class_to_idx}")
            print(f"  Val mapping:   {val_dataset.class_to_idx}")
            
            # Align val_dataset classes and class_to_idx to match train_dataset
            val_dataset.classes = train_dataset.classes
            val_dataset.class_to_idx = train_dataset.class_to_idx
            
            # Re-map target indices in val_dataset.samples and val_dataset.targets
            new_samples = []
            for path, _ in val_dataset.samples:
                # Get the class directory name from the file path
                class_name = Path(path).parent.name
                new_idx = train_dataset.class_to_idx[class_name]
                new_samples.append((path, new_idx))
            val_dataset.samples = new_samples
            val_dataset.targets = [s[1] for s in new_samples]
            print("[SUCCESS] Aligned Val dataset classes and target indices with Train dataset.")
            
        classes = train_dataset.classes
        num_classes = len(classes)
        
        # Weighted Sampler to handle class imbalance
        class_counts = np.array([len([x for x in train_dataset.targets if x == i]) for i in range(num_classes)])
        class_weights = 1.0 / class_counts
        weights = [class_weights[t] for t in train_dataset.targets]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
        
        # D. Get model
        model = get_model(num_classes).to(device)
        
        # E. Loss with weights and label smoothing
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=label_smooth)
        
        # F. Progressive Unfreezing
        unfreeze_epoch = epochs // 3
        
        def set_freeze(model, freeze=True):
            for param in model.parameters():
                param.requires_grad = not freeze
            
            # Always unfreeze classifier head
            for param in model.classifier.parameters():
                param.requires_grad = True
            # For MobileNetV3-Large, keep block 13 un-frozen
            for param in model.features[13:].parameters():
                param.requires_grad = True
                
        # Start training with backbone frozen
        set_freeze(model, True)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0.0
        patience_counter = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # G. Train Loop
        for epoch in range(epochs):
            if epoch == unfreeze_epoch:
                print("Progressive Unfreezing: Unfreezing all backbone layers...")
                set_freeze(model, False)
                optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - unfreeze_epoch)
                
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, mixup_alpha)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc)
            }, step=epoch)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop:
                print(f"Early stopping triggered. No validation accuracy improvement for {early_stop} epochs.")
                break
                
        # H. Finalize model, ONNX, and Meta
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
        
        onnx_path = output_dir / export_cfg.get("onnx_name", "hazelnut_classifier.onnx")
        export_onnx(model, img_size, onnx_path)
        
        # Automatically compile and convert to OpenVINO XML & BIN formats
        try:
            from inference_openvino import convert as convert_ov
            ov_root = Path(export_cfg.get("openvino_dir", "output/openvino"))
            print(f"\nAutomatically converting ONNX model to OpenVINO IR under: {ov_root}...")
            convert_ov(str(onnx_path), str(ov_root))
            print("[SUCCESS] Automatically compiled and exported OpenVINO XML & BIN model.")
        except Exception as ov_err:
            print(f"[WARNING] Automatic OpenVINO compilation failed: {ov_err}")
        
        meta = {
            "classes": classes,
            "img_size": img_size,
            "val_acc": best_acc,
            "history": history,
            "cfg": vars(args)
        }
        meta_path = output_dir / "training_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
        print("Training complete. ONNX model and metadata saved.")
        
        # Log final metrics to MLflow
        mlflow.log_metric("eval_best_val_acc", float(best_acc))
        
        # I. Run Test Set Evaluation & Generate Professional MLOps charts
        print("\n--- Running Evaluation Visualization and Telemetry generation ---")
        try:
            from inference import HazelnutInference
            from evaluate import plot_confusion_matrix, plot_per_class_accuracy, plot_history, save_sample_predictions
            
            engine = HazelnutInference(str(onnx_path), str(meta_path))
            
            y_true = []
            y_pred = []
            
            test_dir = data_dir / "test"
            all_imgs = []
            if test_dir.exists():
                for cls_dir in test_dir.iterdir():
                    if cls_dir.is_dir():
                        imgs = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
                        for img_p in imgs:
                            all_imgs.append((img_p, cls_dir.name))
                            
                for img_p, true_label in all_imgs:
                    res = engine.predict(str(img_p))
                    y_true.append(true_label)
                    y_pred.append(res["class"])
                    
                # Create visual figures in output dir
                cm_img = output_dir / "confusion_matrix.png"
                pca_img = output_dir / "per_class_accuracy.png"
                hist_img = output_dir / "training_history.png"
                samples_img = output_dir / "sample_predictions.png"
                
                plot_confusion_matrix(y_true, y_pred, engine.classes, cm_img)
                plot_per_class_accuracy(y_true, y_pred, engine.classes, pca_img)
                plot_history(history, hist_img)
                save_sample_predictions(engine, data_dir, engine.classes, samples_img)
                
                # Log all evaluation artifacts to MLflow
                if mlflow_cfg.get("log_images", True):
                    mlflow.log_artifact(str(cm_img), "evaluation_charts")
                    mlflow.log_artifact(str(pca_img), "evaluation_charts")
                    mlflow.log_artifact(str(hist_img), "evaluation_charts")
                    mlflow.log_artifact(str(samples_img), "evaluation_charts")
                    print("[SUCCESS] Evaluation charts successfully exported and logged to MLflow.")
            else:
                print("[WARNING] Test split folder not found. Skipping evaluation charts generation.")
        except Exception as eval_err:
            print(f"[WARNING] Post-training evaluation charts generation failed: {eval_err}")
            
        # J. Log the produced models to MLflow Artifacts
        if mlflow_cfg.get("log_models", True):
            print("\nLogging exported model artifacts to MLflow...")
            mlflow.log_artifact(str(onnx_path), "production_models")
            mlflow.log_artifact(str(meta_path), "production_models")
            
            # Check if OpenVINO weights exist in output/openvino and upload them
            ov_root = Path(export_cfg.get("openvino_dir", "output/openvino"))
            if ov_root.exists():
                mlflow.log_artifacts(str(ov_root), "production_models/openvino")
            print("[SUCCESS] All model files correctly logged to MLflow.")
            
    print("\n==============================================================================")
    print("[SUCCESS] End-to-end Advanced Classifier Training Pipeline Complete!")
    print("All parameters, metrics, charts, and model binaries logged to MLflow.")
    print("==============================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Hazelnut Classifier Training")
    parser.add_argument("--config", type=str, default="./config/default.yaml", help="Path to config yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--augment_level", type=str, default=None, choices=["light", "medium", "heavy"])
    parser.add_argument("--label_smooth", type=float, default=None)
    parser.add_argument("--mixup_alpha", type=float, default=None)
    parser.add_argument("--early_stop", type=int, default=None)
    parser.add_argument("--fast_cpu", action="store_true")
    
    args = parser.parse_args()
    train(args)
