import logging
from pathlib import Path
import matplotlib
# Use non-interactive Agg backend to prevent GUI threading crashes in Docker and multithreaded runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path):
    """
    Generates and saves a high-quality annotated confusion matrix heatmap.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear any existing plots to prevent memory leaks
        plt.close("all")

        # Set clean, modern style
        sns.set_theme(style="white")
        
        # Calculate confusion matrix
        cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            square=True,
            cbar_kws={"shrink": 0.8},
            linewidths=1,
            linecolor="#f0f0f0",
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="semibold", labelpad=10)
        ax.set_ylabel("True Label", fontsize=11, fontweight="semibold", labelpad=10)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Successfully generated and saved confusion matrix to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        plt.close("all")


def plot_per_class_accuracy(y_true, y_pred, classes, title, output_path, is_anomaly_detector=False, image_paths=None):
    """
    Generates and saves a per-class accuracy bar chart.
    Supports:
    - Classifier: 4 defect classes (accuracy of multi-class classification).
    - Anomaly Detector: 5 original categories (detection rate: 'good' as Normal, defects as Anomaly).
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.close("all")
        sns.set_theme(style="whitegrid")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        class_accuracies = []
        display_classes = list(classes)

        if is_anomaly_detector and image_paths is not None:
            # For anomaly detector, calculate correct detection per original MVTec folder
            # Normal is correct if pred == 0 (Normal). Defects are correct if pred == 1 (Anomaly).
            image_paths = [Path(p).as_posix() for p in image_paths]
            folder_names = [p.split("/")[-2] for p in image_paths]
            
            display_classes = ["good", "crack", "cut", "hole", "print"]
            for cls in display_classes:
                indices = [i for i, f in enumerate(folder_names) if f.lower() == cls.lower()]
                if not indices:
                    class_accuracies.append(0.0)
                    continue
                
                correct_count = 0
                for idx in indices:
                    true_binary = y_true[idx]
                    pred_binary = y_pred[idx]
                    if cls.lower() == "good":
                        if pred_binary == 0:
                            correct_count += 1
                    else:
                        if pred_binary == 1:
                            correct_count += 1
                
                acc = (correct_count / len(indices)) * 100
                class_accuracies.append(acc)
        else:
            # Multi-class Classifier accuracy per class
            for i, cls in enumerate(classes):
                indices = np.where(y_true == i)[0]
                if len(indices) == 0:
                    class_accuracies.append(0.0)
                    continue
                correct = np.sum(y_pred[indices] == i)
                acc = (correct / len(indices)) * 100
                class_accuracies.append(acc)

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        
        # Color palette: dynamic based on accuracy score
        colors = []
        for acc in class_accuracies:
            if acc >= 90.0:
                colors.append("#2a9d8f")  # Teal green for excellent
            elif acc >= 75.0:
                colors.append("#4a90e2")  # Blue for good
            elif acc >= 50.0:
                colors.append("#f4a261")  # Orange for warning
            else:
                colors.append("#e76f51")  # Coral red for poor

        bars = ax.barh(display_classes, class_accuracies, color=colors, height=0.6, edgecolor="none")
        
        # Add labels to the ends of the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#404040"
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Accuracy Percentage (%)", fontsize=10, fontweight="semibold", labelpad=10)
        ax.set_xlim(0, 112)  # Leave room for labels
        
        # Style clean-up
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Successfully generated and saved per-class accuracy chart to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting per-class accuracy: {e}")
        plt.close("all")
