import os
import sqlite3
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
from PIL import Image
from mlflow.tracking import MlflowClient


class MlflowTracker:
    def __init__(self, config: dict):
        self.config = config
        self.run = None
        self.client = None

    def start(self):
        mlflow_cfg = self.config.get("mlflow", {})
        if not mlflow_cfg:
            return None

        tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        experiment_name = mlflow_cfg.get(
            "experiment_name", "WeirdHazelnut_Integrated_Pipeline"
        )
        artifact_root = mlflow_cfg.get("artifact_root")

        _repair_sqlite_artifact_locations(tracking_uri, experiment_name, artifact_root)

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        _ensure_experiment(self.client, experiment_name, artifact_root)
        mlflow.set_experiment(experiment_name)

        try:
            mlflow.enable_system_metrics_logging()
        except Exception as e:
            print(f"Failed to enable system metrics: {e}")

        run_name = f"{mlflow_cfg.get('run_name', 'API_Session')}_{int(time.time())}"
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            {
                "ad_model": self.config["models"]["anomaly_detector"]["path"],
                "cls_model": self.config["models"]["classifier"]["model_path"],
                "threshold_low": self.config["pipeline"]["thresholds"]["low"],
                "threshold_high": self.config["pipeline"]["thresholds"]["high"],
            }
        )
        print(f"MLflow tracking initialized: {run_name}")
        return self.run

    def end(self):
        if self.run:
            mlflow.end_run()
            self.run = None

    def log_prediction(self, image: Image.Image, result: dict, total_latency_ms: float):
        if not self.run or not self.client:
            return

        try:
            heat_map = result.get("heat_map")
            metrics = {
                "anomaly_score": result["anomaly_score"],
                "ad_latency_ms": result["ad_latency_ms"],
                "classifier_latency_ms": result["classifier_latency_ms"],
                "total_latency_ms": total_latency_ms,
                "is_anomaly": 1.0 if result["anomaly"] else 0.0,
                "uncertain": 1.0 if result["uncertain"] else 0.0,
            }
            if result.get("label"):
                metrics["confidence"] = result.get("confidence", 0.0)

            run_id = self.run.info.run_id
            for key, value in metrics.items():
                self.client.log_metric(run_id, key, float(value))

            if not self.config.get("mlflow", {}).get("log_images", False):
                return

            self._log_prediction_images(run_id, image, heat_map, result)
        except Exception as e:
            print(f"MLflow logging error: {e}")

    def _log_prediction_images(self, run_id: str, image: Image.Image, heat_map, result: dict):
        ts = int(time.time() * 1000)
        label = result.get("label") or ("uncertain" if result.get("uncertain") else "normal")
        filename = f"{ts}_{label}_original_heatmap.png"

        with tempfile.TemporaryDirectory() as tmpdir:
            original = image.convert("RGB").resize((256, 256))
            if heat_map is not None:
                heatmap_img = _array_to_rgb_image(heat_map).resize((256, 256))
            else:
                heatmap_img = Image.new("RGB", (256, 256), color=(0, 0, 0))

            collage = Image.new("RGB", (512, 256))
            collage.paste(original, (0, 0))
            collage.paste(heatmap_img, (256, 0))
            collage_path = os.path.join(tmpdir, filename)
            collage.save(collage_path)
            self.client.log_artifact(run_id, collage_path, artifact_path="predictions")


def _array_to_rgb_image(array) -> Image.Image:
    arr = np.asarray(array)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unsupported heatmap shape for MLflow artifact logging: {arr.shape}")


def resolve_model_paths(config: dict) -> tuple[str, str, str]:
    ad_model_path = config["models"]["anomaly_detector"]["path"]
    cls_model_path = config["models"]["classifier"]["model_path"]
    cls_meta_path = config["models"]["classifier"]["meta_path"]

    reg_cfg = config.get("registry", {})
    if not reg_cfg.get("enabled", False):
        return ad_model_path, cls_model_path, cls_meta_path

    print("Fetching models from MLflow Registry...")
    try:
        ad_local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=reg_cfg.get("anomaly_detector")
        )
        cls_local_dir = mlflow.artifacts.download_artifacts(artifact_uri=reg_cfg.get("classifier"))
        return (
            os.path.join(ad_local_dir, "model.xml"),
            os.path.join(cls_local_dir, "hazelnut_classifier.onnx"),
            os.path.join(cls_local_dir, "training_meta.json"),
        )
    except Exception as e:
        print(f"Failed to fetch models from registry: {e}. Falling back to local models.")
        return ad_model_path, cls_model_path, cls_meta_path


def _ensure_experiment(client: MlflowClient, experiment_name: str, artifact_root: str | None):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment

    if artifact_root:
        return client.create_experiment(experiment_name, artifact_location=artifact_root)
    return client.create_experiment(experiment_name)


def _repair_sqlite_artifact_locations(
    tracking_uri: str,
    experiment_name: str,
    artifact_root: str | None,
):
    """Repair stale host artifact URIs when Docker uses a mounted MLflow DB.

    MLflow stores artifact paths in SQLite. If the DB was first created on
    Windows, Docker cannot read file:///D:/... paths from the MLflow UI.
    """
    if not artifact_root or not tracking_uri.startswith("sqlite:///"):
        return

    db_path = _sqlite_path_from_uri(tracking_uri)
    if not db_path or not db_path.exists():
        return

    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        row = cur.execute(
            "select experiment_id, artifact_location from experiments where name = ?",
            (experiment_name,),
        ).fetchone()
        if not row:
            con.close()
            return

        experiment_id = str(row[0])
        expected_experiment_uri = _join_artifact_uri(artifact_root, experiment_id)
        if row[1] != expected_experiment_uri:
            cur.execute(
                "update experiments set artifact_location = ? where experiment_id = ?",
                (expected_experiment_uri, row[0]),
            )

        runs = cur.execute(
            "select run_uuid, artifact_uri from runs where experiment_id = ?",
            (row[0],),
        ).fetchall()
        for run_uuid, artifact_uri in runs:
            expected_run_uri = _join_artifact_uri(expected_experiment_uri, run_uuid, "artifacts")
            if artifact_uri != expected_run_uri:
                cur.execute(
                    "update runs set artifact_uri = ? where run_uuid = ?",
                    (expected_run_uri, run_uuid),
                )

        con.commit()
        con.close()
    except Exception as e:
        print(f"MLflow artifact URI repair skipped: {e}")


def _sqlite_path_from_uri(uri: str) -> Path | None:
    raw_path = uri.removeprefix("sqlite:///")
    if raw_path.startswith("/"):
        return Path(raw_path)
    return Path(raw_path)


def _join_artifact_uri(*parts: str) -> str:
    cleaned = [str(part).strip("/") for part in parts if str(part)]
    if not cleaned:
        return ""

    first = cleaned[0]
    if "://" in first:
        parsed = urlparse(first)
        base_path = parsed.path.rstrip("/")
        suffix = "/".join(cleaned[1:])
        path = f"{base_path}/{suffix}" if suffix else base_path
        return f"{parsed.scheme}://{path}"

    return "/".join(cleaned)
