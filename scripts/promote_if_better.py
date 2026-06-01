from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from weird_hazelnut.config import load_config


def find_latest_file(root: str | Path, pattern: str) -> Path:
    root = Path(root)
    candidates = list(root.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No file found: {root}/**/{pattern}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")


def get_latest_run_by_name(
    client: MlflowClient,
    experiment_id: str,
    run_name: str,
    before_start_time: int | None = None,
):
    filter_parts = [f"tags.mlflow.runName = '{run_name}'"]

    if before_start_time is not None:
        filter_parts.append(f"attributes.start_time < {before_start_time}")

    filter_string = " and ".join(filter_parts)

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )

    return runs[0] if runs else None


def promote_candidate_artifacts(config: dict) -> dict:
    ad_training_cfg = config.get("training", {}).get("anomaly_detector", {})
    cls_training_cfg = config.get("training", {}).get("classifier", {})

    ad_prod_xml = Path(config["models"]["anomaly_detector"]["path"])
    ad_prod_bin = ad_prod_xml.with_suffix(".bin")

    cls_prod_onnx = Path(config["models"]["classifier"]["model_path"])
    cls_prod_meta = Path(config["models"]["classifier"]["meta_path"])

    ad_export_root = ad_training_cfg.get(
        "export_root",
        "models/anomaly_detector_retrained",
    )
    cls_output_dir = Path(
        cls_training_cfg.get(
            "output_dir",
            "models/classifier_retrained",
        )
    )

    candidate_ad_xml = find_latest_file(ad_export_root, "model.xml")
    candidate_ad_bin = candidate_ad_xml.with_suffix(".bin")

    candidate_cls_onnx = cls_output_dir / cls_training_cfg.get("onnx_name", "model.onnx")
    candidate_cls_meta = cls_output_dir / cls_training_cfg.get("meta_name", "meta.json")

    copy_file(candidate_ad_xml, ad_prod_xml)
    copy_file(candidate_ad_bin, ad_prod_bin)
    copy_file(candidate_cls_onnx, cls_prod_onnx)
    copy_file(candidate_cls_meta, cls_prod_meta)

    return {
        "promoted_ad_xml": str(ad_prod_xml),
        "promoted_ad_bin": str(ad_prod_bin),
        "promoted_cls_onnx": str(cls_prod_onnx),
        "promoted_cls_meta": str(cls_prod_meta),
        "candidate_ad_xml": str(candidate_ad_xml),
        "candidate_ad_bin": str(candidate_ad_bin),
        "candidate_cls_onnx": str(candidate_cls_onnx),
        "candidate_cls_meta": str(candidate_cls_meta),
    }


def main() -> None:
    config = load_config()

    mlflow_cfg = config.get("mlflow", {})
    orchestration_cfg = config.get("orchestration", {})
    metric_gate_cfg = orchestration_cfg.get("metric_gate", {})

    metric_name = metric_gate_cfg.get("primary_metric", "eval_system_accuracy")
    min_improvement = float(metric_gate_cfg.get("min_improvement", 0.01))

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(
        mlflow_cfg.get("experiment_name", "WeirdHazelnut_Integrated_Pipeline")
    )

    client = MlflowClient()
    experiment = client.get_experiment_by_name(
        mlflow_cfg.get("experiment_name", "WeirdHazelnut_Integrated_Pipeline")
    )

    if not experiment:
        raise RuntimeError("MLflow experiment not found.")

    candidate_run = get_latest_run_by_name(
        client=client,
        experiment_id=experiment.experiment_id,
        run_name="Candidate_Model_Evaluation",
    )

    if not candidate_run:
        raise RuntimeError("No Candidate_Model_Evaluation run found.")

    baseline_run = get_latest_run_by_name(
        client=client,
        experiment_id=experiment.experiment_id,
        run_name="Full_Test_Set_Evaluation",
        before_start_time=candidate_run.info.start_time,
    )

    candidate_metric = candidate_run.data.metrics.get(metric_name)

    if candidate_metric is None:
        raise RuntimeError(
            f"Candidate run does not contain metric '{metric_name}'. "
            f"Run ID: {candidate_run.info.run_id}"
        )

    baseline_metric = None
    if baseline_run:
        baseline_metric = baseline_run.data.metrics.get(metric_name)

    if baseline_metric is None:
        improvement = None
        should_promote = True
        decision_reason = "No valid baseline metric found. Promoting first candidate."
    else:
        improvement = candidate_metric - baseline_metric
        should_promote = improvement >= min_improvement
        decision_reason = (
            "Candidate improved enough."
            if should_promote
            else "Candidate did not improve enough."
        )

    promoted_artifacts = {}

    if should_promote:
        promoted_artifacts = promote_candidate_artifacts(config)

    with mlflow.start_run(run_name="Model_Promotion_Decision") as run:
        mlflow.log_params(
            {
                "metric_name": metric_name,
                "min_improvement": min_improvement,
                "candidate_run_id": candidate_run.info.run_id,
                "baseline_run_id": baseline_run.info.run_id if baseline_run else None,
                "promotion_decision": "accepted" if should_promote else "rejected",
                "decision_reason": decision_reason,
            }
        )

        mlflow.log_metrics(
            {
                "candidate_metric": float(candidate_metric),
                "baseline_metric": float(baseline_metric)
                if baseline_metric is not None
                else float("nan"),
                "improvement": float(improvement)
                if improvement is not None
                else float("nan"),
            }
        )

        for key, value in promoted_artifacts.items():
            mlflow.log_param(key, value)

    print("Promotion decision:")
    print(f"  metric_name: {metric_name}")
    print(f"  candidate_metric: {candidate_metric}")
    print(f"  baseline_metric: {baseline_metric}")
    print(f"  improvement: {improvement}")
    print(f"  min_improvement: {min_improvement}")
    print(f"  decision: {'accepted' if should_promote else 'rejected'}")
    print(f"  reason: {decision_reason}")


if __name__ == "__main__":
    main()