from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer
from weird_hazelnut.data.sync_worker import DataSyncWorker
from weird_hazelnut.training.retrain import register_latest_models, run_retraining

logger = logging.getLogger(__name__)

LABELS = ["good", "crack", "cut", "hole", "print"]
DEFECT_LABELS = ["crack", "cut", "hole", "print"]


def sync_labels() -> None:
    worker = DataSyncWorker()
    worker.sync()


def dataset_snapshot() -> dict:
    config = load_config()
    data_layer = create_data_layer(config)
    if not data_layer:
        raise RuntimeError("data_layer.enabled must be true.")

    counts = {}
    for split in ["train", "val", "test"]:
        for label in LABELS:
            counts[f"{split}/{label}"] = len(
                data_layer.list_dataset_images(split=split, labels=[label])
            )

    counts["train/total"] = sum(counts[f"train/{label}"] for label in LABELS)
    counts["train/defect_total"] = sum(counts[f"train/{label}"] for label in DEFECT_LABELS)
    counts["val/total"] = sum(counts[f"val/{label}"] for label in LABELS)
    counts["test/total"] = sum(counts[f"test/{label}"] for label in LABELS)

    return counts


def data_gate(counts: dict) -> dict:
    config = load_config()
    gate_cfg = config.get("orchestration", {}).get("retrain_gate", {})

    min_train_total = int(gate_cfg.get("min_train_total", 150))
    min_train_defect_total = int(gate_cfg.get("min_train_defect_total", 10))
    min_val_total = int(gate_cfg.get("min_val_total", 3))
    min_test_total = int(gate_cfg.get("min_test_total", 10))

    reasons = []

    if counts["train/total"] < min_train_total:
        reasons.append(f"train/total={counts['train/total']} < {min_train_total}")

    if counts["train/defect_total"] < min_train_defect_total:
        reasons.append(
            f"train/defect_total={counts['train/defect_total']} < {min_train_defect_total}"
        )

    if counts["val/total"] < min_val_total:
        reasons.append(f"val/total={counts['val/total']} < {min_val_total}")

    if counts["test/total"] < min_test_total:
        reasons.append(f"test/total={counts['test/total']} < {min_test_total}")

    result = {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "counts": counts,
    }

    if not result["passed"]:
        raise RuntimeError(f"Data gate failed: {result}")

    return result


def get_baseline_metric() -> dict:
    config = load_config()
    mlflow_cfg = config.get("mlflow", {})
    metric_name = (
        config.get("orchestration", {})
        .get("metric_gate", {})
        .get("primary_metric", "eval_system_accuracy")
    )

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        mlflow_cfg.get("experiment_name", "WeirdHazelnut_Integrated_Pipeline")
    )

    if not experiment:
        return {"found": False, "metric_name": metric_name, "value": None, "run_id": None}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric_name} >= 0",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )

    if not runs:
        return {"found": False, "metric_name": metric_name, "value": None, "run_id": None}

    run = runs[0]
    return {
        "found": True,
        "metric_name": metric_name,
        "value": run.data.metrics.get(metric_name),
        "run_id": run.info.run_id,
    }


def retrain_models() -> dict:
    config = load_config()
    data_layer = create_data_layer(config)
    if not data_layer:
        raise RuntimeError("data_layer.enabled must be true.")

    result = run_retraining(config, data_layer)

    if not result["success"]:
        raise RuntimeError(f"Retraining failed: {result}")

    return result


def _find_latest_file(root: str | Path, pattern: str) -> str | None:
    root = Path(root)
    if not root.exists():
        return None

    candidates = list(root.rglob(pattern))
    if not candidates:
        return None

    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def build_candidate_model_paths() -> dict:
    config = load_config()

    ad_cfg = config.get("training", {}).get("anomaly_detector", {})
    cls_cfg = config.get("training", {}).get("classifier", {})

    ad_export_root = ad_cfg.get("export_root", "models/anomaly_detector_retrained")
    cls_output_dir = Path(cls_cfg.get("output_dir", "models/classifier_retrained"))

    anomaly_model_path = _find_latest_file(ad_export_root, "model.xml")
    classifier_model_path = cls_output_dir / cls_cfg.get("onnx_name", "model.onnx")
    classifier_meta_path = cls_output_dir / cls_cfg.get("meta_name", "meta.json")

    missing = []

    if not anomaly_model_path:
        missing.append(f"{ad_export_root}/**/model.xml")
    if not classifier_model_path.exists():
        missing.append(str(classifier_model_path))
    if not classifier_meta_path.exists():
        missing.append(str(classifier_meta_path))

    if missing:
        raise FileNotFoundError("Candidate model artifact(s) not found: " + ", ".join(missing))

    return {
        "anomaly_model_path": anomaly_model_path,
        "classifier_model_path": str(classifier_model_path),
        "classifier_meta_path": str(classifier_meta_path),
    }


def evaluate_candidate(candidate_paths: dict) -> dict:
    config = load_config()
    from weird_hazelnut.evaluation.pipeline_eval import evaluate

    return evaluate(
        config=config,
        model_overrides=candidate_paths,
        run_name="Candidate_Model_Evaluation",
    )


def metric_gate(baseline: dict, candidate_metrics: dict) -> dict:
    config = load_config()
    gate_cfg = config.get("orchestration", {}).get("metric_gate", {})

    metric_name = gate_cfg.get("primary_metric", "eval_system_accuracy")
    min_improvement = float(gate_cfg.get("min_improvement", 0.01))

    candidate_value = candidate_metrics.get(metric_name)

    if candidate_value is None:
        raise RuntimeError(f"Candidate metric {metric_name} not found: {candidate_metrics}")

    if not baseline.get("found"):
        return {
            "passed": True,
            "reason": "No baseline evaluation found. Accepting first candidate.",
            "baseline": baseline,
            "candidate_value": candidate_value,
        }

    baseline_value = baseline["value"]
    improvement = candidate_value - baseline_value
    passed = improvement >= min_improvement

    result = {
        "passed": passed,
        "metric_name": metric_name,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "improvement": improvement,
        "min_improvement": min_improvement,
    }

    if not passed:
        raise RuntimeError(f"Metric gate failed: {result}")

    return result


def register_models() -> dict:
    config = load_config()

    if not config.get("registry", {}).get("enabled", False):
        return {
            "registered": False,
            "reason": "registry.enabled=false",
        }

    register_latest_models(config["mlflow"]["experiment_name"])

    return {
        "registered": True,
        "reason": "Latest retrained models registered.",
    }