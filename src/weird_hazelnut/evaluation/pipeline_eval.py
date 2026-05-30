import time
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image
from tqdm import tqdm

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer
from weird_hazelnut.pipeline import HazelnutPipeline


def evaluate():
    config = load_config()
    data_layer = create_data_layer(config)
    pipeline = HazelnutPipeline(
        anomaly_model_path=config["models"]["anomaly_detector"]["path"],
        classifier_model_path=config["models"]["classifier"]["model_path"],
        classifier_meta_path=config["models"]["classifier"]["meta_path"],
        threshold_low=config["pipeline"]["thresholds"]["low"],
        threshold_high=config["pipeline"]["thresholds"]["high"],
        lake_dir=config["pipeline"]["lake_dir"],
        anomaly_device=config["models"]["anomaly_detector"].get("device", "CPU"),
        data_layer=data_layer,
    )

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(
        mlflow_cfg.get("experiment_name", "WeirdHazelnut_Integrated_Pipeline")
    )

    with mlflow.start_run(run_name="Full_Test_Set_Evaluation") as run:
        mlflow.log_params(
            {
                "ad_model": config["models"]["anomaly_detector"]["path"],
                "cls_model": config["models"]["classifier"]["model_path"],
                "threshold_low": config["pipeline"]["thresholds"]["low"],
                "threshold_high": config["pipeline"]["thresholds"]["high"],
            }
        )

        results = []
        latencies = []

        print("Starting evaluation...")
        for image, true_label in tqdm(_iter_test_images(data_layer)):
            is_truly_anomalous = true_label != "good"

            start_time = time.perf_counter()
            res = pipeline.run(image, persist=False)
            latency = (time.perf_counter() - start_time) * 1000

            res["true_label"] = true_label
            res["is_truly_anomalous"] = is_truly_anomalous
            res["latency"] = latency

            results.append(res)
            latencies.append(latency)

        if not results:
            print("No test samples found. Run scripts/migrate_dataset_to_datalake.py first.")
            return

        total = len(results)
        ad_accuracy = sum(
            1 for r in results if r["anomaly"] == r["is_truly_anomalous"]
        ) / total

        true_anomalies_detected = [
            r for r in results if r["is_truly_anomalous"] and r["anomaly"]
        ]
        cls_accuracy = 0.0
        if true_anomalies_detected:
            cls_accuracy = sum(
                1 for r in true_anomalies_detected if r["label"] == r["true_label"]
            ) / len(true_anomalies_detected)

        system_correct = 0
        for r in results:
            if not r["is_truly_anomalous"] and not r["anomaly"]:
                system_correct += 1
            elif r["is_truly_anomalous"] and r["anomaly"] and r["label"] == r["true_label"]:
                system_correct += 1

        metrics = {
            "test_total_samples": float(total),
            "eval_ad_accuracy": float(ad_accuracy),
            "eval_cls_accuracy": float(cls_accuracy),
            "eval_system_accuracy": float(system_correct / total),
            "eval_avg_latency_ms": float(np.mean(latencies)),
            "eval_p95_latency_ms": float(np.percentile(latencies, 95)),
        }
        mlflow.log_metrics(metrics)

        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        experiment = mlflow.get_experiment(run.info.experiment_id)
        print(f"\nSuccessfully logged evaluation to MLflow experiment: {experiment.name}")

def _iter_test_images(data_layer):
    if data_layer:
        for item, image_record in data_layer.list_dataset_images(split="test"):
            yield data_layer.open_image(image_record.minio_object_key), item.label
        return

    test_dir = Path("data/hazelnut/test")
    if not test_dir.exists():
        print(f"Test directory not found at {test_dir}")
        return

    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.png"):
            yield Image.open(img_path).convert("RGB"), class_dir.name


if __name__ == "__main__":
    evaluate()
