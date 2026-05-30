import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from tqdm import tqdm

from inference import HazelnutInference

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = sk_confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_per_class_accuracy(y_true, y_pred, classes, output_path):
    accs = []
    for cls in classes:
        idxs = [i for i, label in enumerate(y_true) if label == cls]
        if not idxs:
            accs.append(0)
            continue
        correct = sum(1 for i in idxs if y_pred[i] == cls)
        accs.append(correct / len(idxs))
        
    colors = []
    for acc in accs:
        if acc >= 0.90: colors.append('green')
        elif acc >= 0.75: colors.append('orange')
        else: colors.append('red')
        
    plt.figure(figsize=(10, 6))
    plt.bar(classes, [a * 100 for a in accs], color=colors)
    plt.axhline(y=90, color='gray', linestyle='--')
    plt.axhline(y=75, color='gray', linestyle='--')
    plt.title("Per-Class Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.savefig(output_path)
    plt.close()

def plot_history(history, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.savefig(output_path)
    plt.close()

def save_sample_predictions(engine, data_dir, classes, output_path):
    samples = []
    for cls in classes:
        cls_dir = data_dir / "test" / cls
        if not cls_dir.exists(): continue
        imgs = list(cls_dir.glob("*.png"))[:4]
        for img_p in imgs:
            res = engine.predict(str(img_p))
            samples.append({
                "path": img_p,
                "true": cls,
                "pred": res["class"],
                "conf": res["confidence"]
            })
            
    if not samples: return
    
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples):
        img = Image.open(sample["path"])
        axes[i].imshow(img)
        color = 'green' if sample["true"] == sample["pred"] else 'red'
        axes[i].set_title(f"T: {sample['true']}\nP: {sample['pred']} ({sample['conf']:.2f})", color=color)
        axes[i].axis('off')
        # Add border
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate(args):
    engine = HazelnutInference(args.onnx, args.meta)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = []
    y_pred = []
    
    test_dir = data_dir / "test"
    all_imgs = []
    for cls_dir in test_dir.iterdir():
        if cls_dir.is_dir():
            imgs = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
            for img_p in imgs:
                all_imgs.append((img_p, cls_dir.name))
                
    for img_p, true_label in tqdm(all_imgs, desc="Evaluating"):
        res = engine.predict(str(img_p))
        y_true.append(true_label)
        y_pred.append(res["class"])
        
    plot_confusion_matrix(y_true, y_pred, engine.classes, output_dir / "confusion_matrix.png")
    plot_per_class_accuracy(y_true, y_pred, engine.classes, output_dir / "per_class_accuracy.png")
    plot_history(engine.meta["history"], output_dir / "training_history.png")
    save_sample_predictions(engine, data_dir, engine.classes, output_dir / "sample_predictions.png")
    
    print(f"Evaluation complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hazelnut Classifier Evaluation")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/")
    
    args = parser.parse_args()
    evaluate(args)
