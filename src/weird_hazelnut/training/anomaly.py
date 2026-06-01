from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np

from weird_hazelnut.data.layer import DataLayer
from weird_hazelnut.training.minio_dataset import (
    AnomalibFolderStaging,
    dataset_counts,
    detect_leakage,
    load_samples,
)

logger = logging.getLogger(__name__)


def create_anomalib_model(config: dict):
    from anomalib.models import Padim, Patchcore

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    model_name = model_cfg.get("name", "patchcore").lower()
    image_size = tuple(data_cfg.get("image_size", [256, 256]))

    if model_name == "patchcore":
        pre_processor = Patchcore.configure_pre_processor(image_size=image_size)
        return Patchcore(
            backbone=model_cfg.get("backbone", "resnet18"),
            coreset_sampling_ratio=model_cfg.get("coreset_sampling_ratio", 0.005),
            num_neighbors=model_cfg.get("num_neighbors", 9),
            pre_processor=pre_processor,
        )
    if model_name == "padim":
        pre_processor = Padim.configure_pre_processor(image_size=image_size)
        return Padim(
            backbone=model_cfg.get("backbone", "resnet18"),
            layers=model_cfg.get("layers", ["layer1", "layer2", "layer3"]),
            pre_processor=pre_processor,
        )
    raise ValueError(f"Unsupported anomaly model: {model_name}")


def create_datamodule(config: dict, root: Path):
    from anomalib.data import Folder

    data_cfg = config.get("data", {})
    return Folder(
        name=data_cfg.get("name", "hazelnut"),
        root=str(root),
        normal_dir=data_cfg.get("normal_dir", "train/good"),
        normal_test_dir=data_cfg.get("normal_test_dir", "test/good"),
        abnormal_dir=data_cfg.get(
            "abnormal_dir",
            ["test/crack", "test/cut", "test/hole", "test/print"],
        ),
        train_batch_size=data_cfg.get("train_batch_size", 32),
        eval_batch_size=data_cfg.get("test_batch_size", 32),
        num_workers=data_cfg.get("num_workers", 0),
    )


def train_anomaly_detector(config: dict, data_layer: DataLayer) -> bool:
    try:
        from anomalib.deploy import ExportType
        from anomalib.engine import Engine
        from anomalib.loggers import AnomalibMLFlowLogger
    except Exception as exc:
        logger.error("Anomalib is required for anomaly retraining: %s", exc)
        return False

    training_cfg = config.get("training", {}).get("anomaly_detector", {})
    mlflow_cfg = config.get("mlflow", {})
    model_cfg = {
        "data": {
            "name": "hazelnut",
            "normal_dir": "train/good",
            "normal_test_dir": "test/good",
            "abnormal_dir": training_cfg.get(
                "abnormal_dir",
                ["test/crack", "test/cut", "test/hole", "test/print"],
            ),
            "image_size": training_cfg.get("image_size", [256, 256]),
            "train_batch_size": training_cfg.get("train_batch_size", 32),
            "test_batch_size": training_cfg.get("test_batch_size", 32),
            "num_workers": training_cfg.get("num_workers", 0),
        },
        "model": {
            "name": training_cfg.get("model_name", "patchcore"),
            "backbone": training_cfg.get("backbone", "resnet18"),
            "coreset_sampling_ratio": training_cfg.get("coreset_sampling_ratio", 0.005),
            "num_neighbors": training_cfg.get("num_neighbors", 9),
            "layers": training_cfg.get("layers", ["layer1", "layer2", "layer3"]),
        },
        "engine": {
            "max_epochs": training_cfg.get("max_epochs", 3),
            "devices": training_cfg.get("devices", 0),
            "accelerator": training_cfg.get("accelerator", "cpu"),
        },
        "export": {
            "export_type": training_cfg.get("export_type", "OPENVINO"),
            "export_root": training_cfg.get("export_root", "models/anomaly_detector_retrained"),
        },
    }

    all_samples = load_samples(data_layer)
    train_good_count = len(load_samples(data_layer, split="train", labels=["good"]))
    test_count = len(load_samples(data_layer, split="test"))
    if train_good_count == 0 or test_count == 0:
        logger.error("Anomaly retraining needs train/good and test samples in MinIO/Postgres.")
        return False

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "WeirdHazelnut_Integrated_Pipeline"))
    run_name = training_cfg.get("run_name", "Retraining_AD_MinIO")

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
            from weird_hazelnut.data.repositories import DataRepository
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
                "trainer": "integrated_anomalib",
                "data_source": "minio_postgres",
                "model_name": model_cfg["model"]["name"],
                "model_backbone": model_cfg["model"]["backbone"],
                "engine_max_epochs": model_cfg["engine"]["max_epochs"],
                "export_type": model_cfg["export"]["export_type"],
                "data_leakage_detected": bool(duplicates),
                "data_leakage_count": len(duplicates),
            }
        )
        mlflow.log_params(dataset_counts(all_samples))

        anomalib_logger = AnomalibMLFlowLogger(
            tracking_uri=mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"),
            experiment_name=mlflow_cfg.get("experiment_name", "WeirdHazelnut_Integrated_Pipeline"),
            run_id=run.info.run_id,
        )
        with AnomalibFolderStaging(data_layer) as dataset_root:
            datamodule = create_datamodule(model_cfg, dataset_root)
            model = create_anomalib_model(model_cfg)
            engine = Engine(
                max_epochs=model_cfg["engine"]["max_epochs"],
                devices=model_cfg["engine"]["devices"],
                accelerator=model_cfg["engine"]["accelerator"],
                logger=anomalib_logger,
            )
            engine.fit(model=model, datamodule=datamodule)
            test_results = engine.test(model=model, datamodule=datamodule)
            for result in test_results or []:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"ad_{key}", float(value))

            # Log optimal calculated anomaly threshold
            try:
                if hasattr(model, "image_threshold"):
                    threshold_val = float(model.image_threshold.value.item()) if hasattr(model.image_threshold, "value") else float(model.image_threshold)
                    mlflow.log_metric("ad_calculated_image_threshold", threshold_val)
                    logger.info(f"Successfully retrieved and logged ad_calculated_image_threshold: {threshold_val}")
                elif test_results and "image_threshold" in test_results[0]:
                    mlflow.log_metric("ad_calculated_image_threshold", float(test_results[0]["image_threshold"]))
            except Exception as thresh_err:
                logger.warning(f"Could not retrieve optimal anomaly threshold: {thresh_err}")

            # Collect predictions for MLflow evaluation plots
            y_true_binary = []
            y_pred_binary = []
            paths_list = []
            try:
                logger.info("Generating anomaly detector predictions for MLflow evaluation...")
                predictions = engine.predict(model=model, datamodule=datamodule)
                
                if not predictions:
                    logger.warning("No predictions returned by engine.predict()!")
                else:
                    logger.info(f"Received {len(predictions)} batches of predictions.")
                    first_batch = predictions[0]
                    try:
                        if hasattr(first_batch, "keys"):
                            logger.info(f"First prediction batch keys: {list(first_batch.keys())}")
                        elif hasattr(first_batch, "__dict__"):
                            logger.info(f"First prediction batch dict keys: {list(first_batch.__dict__.keys())}")
                        else:
                            logger.info(f"First prediction batch type: {type(first_batch)}")
                    except Exception as logging_err:
                        logger.warning(f"Could not log batch structure: {logging_err}")

                import torch
                for batch in predictions or []:
                    if hasattr(batch, "get"):
                        batch_labels = batch.get("gt_label") if "gt_label" in batch else (
                            batch.get("gt_labels") if "gt_labels" in batch else (
                                batch.get("label") if "label" in batch else batch.get("labels")
                            )
                        )
                        batch_preds = batch.get("pred_label") if "pred_label" in batch else batch.get("pred_labels")
                        batch_paths = batch.get("image_path") if "image_path" in batch else batch.get("image_paths")
                    else:
                        batch_labels = getattr(batch, "gt_label", getattr(batch, "gt_labels", getattr(batch, "label", getattr(batch, "labels", None))))
                        batch_preds = getattr(batch, "pred_label", getattr(batch, "pred_labels", None))
                        batch_paths = getattr(batch, "image_path", getattr(batch, "image_paths", None))

                    if isinstance(batch_labels, torch.Tensor):
                        batch_labels = batch_labels.cpu().numpy()
                    elif hasattr(batch_labels, "cpu"):
                        batch_labels = batch_labels.cpu().numpy()

                    if isinstance(batch_preds, torch.Tensor):
                        batch_preds = batch_preds.cpu().numpy()
                    elif hasattr(batch_preds, "cpu"):
                        batch_preds = batch_preds.cpu().numpy()

                    # Safely collect true binary labels
                    if isinstance(batch_labels, (list, np.ndarray)) or (hasattr(batch_labels, "__len__") and not isinstance(batch_labels, str)):
                        y_true_binary.extend(list(batch_labels))
                    elif batch_labels is not None:
                        y_true_binary.append(batch_labels)

                    # Safely collect predicted binary labels
                    if isinstance(batch_preds, (list, np.ndarray)) or (hasattr(batch_preds, "__len__") and not isinstance(batch_preds, str)):
                        y_pred_binary.extend(list(batch_preds))
                    elif batch_preds is not None:
                        y_pred_binary.append(batch_preds)

                    # Safely collect image file paths
                    if isinstance(batch_paths, str):
                        paths_list.append(batch_paths)
                    elif isinstance(batch_paths, (list, np.ndarray)) or hasattr(batch_paths, "__len__"):
                        paths_list.extend(list(batch_paths))
                    elif batch_paths is not None:
                        paths_list.append(batch_paths)
                
                logger.info(f"Extracted {len(y_true_binary)} true labels, {len(y_pred_binary)} predicted labels, and {len(paths_list)} paths.")
            except Exception as e:
                logger.error(f"Failed to generate anomaly detector predictions for plots: {e}", exc_info=True)

            export_root = Path(model_cfg["export"]["export_root"])
            cm_path = export_root / "anomaly_confusion_matrix.png"
            acc_path = export_root / "anomaly_per_class_accuracy.png"

            if len(y_true_binary) > 0 and len(y_pred_binary) > 0:
                from weird_hazelnut.training.visualization import plot_confusion_matrix, plot_per_class_accuracy
                
                plot_confusion_matrix(
                    y_true=y_true_binary,
                    y_pred=y_pred_binary,
                    classes=["Normal", "Anomaly"],
                    title="Anomaly Detector Confusion Matrix",
                    output_path=cm_path
                )
                
                plot_per_class_accuracy(
                    y_true=y_true_binary,
                    y_pred=y_pred_binary,
                    classes=["good", "crack", "cut", "hole", "print"],
                    title="Anomaly Detector Per-Class Accuracy",
                    output_path=acc_path,
                    is_anomaly_detector=True,
                    image_paths=paths_list
                )

            export_type = ExportType[model_cfg["export"]["export_type"].upper()]
            export_path = engine.export(
                model=model,
                export_type=export_type,
                export_root=model_cfg["export"]["export_root"],
                input_size=model_cfg["data"]["image_size"],
                datamodule=datamodule,
            )
            if export_path and Path(export_path).exists():
                mlflow.log_artifacts(str(Path(export_path).parent), "production_models")
                
                # Automated Latency Benchmark for exported OpenVINO model
                try:
                    logger.info("Benchmarking OpenVINO anomaly detector model latency...")
                    import openvino as ov
                    import time
                    core = ov.Core()
                    ov_model_xml = Path(export_path).parent / "openvino" / "model.xml"
                    if ov_model_xml.exists():
                        ov_model = core.read_model(model=str(ov_model_xml))
                        compiled_model = core.compile_model(ov_model, "CPU")
                        infer_request = compiled_model.create_infer_request()
                        
                        input_size = model_cfg["data"]["image_size"] # e.g. [256, 256]
                        dummy_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)
                        
                        # Warmup
                        for _ in range(5):
                            infer_request.infer([dummy_input])
                            
                        # Benchmark
                        latencies = []
                        for _ in range(30):
                            t0 = time.perf_counter()
                            infer_request.infer([dummy_input])
                            latencies.append((time.perf_counter() - t0) * 1000.0)
                            
                        avg_latency = float(np.mean(latencies))
                        p95_latency = float(np.percentile(latencies, 95))
                        mlflow.log_metric("ad_inference_latency_avg_ms", avg_latency)
                        mlflow.log_metric("ad_inference_latency_p95_ms", p95_latency)
                        logger.info(f"OpenVINO inference latency logged: Avg {avg_latency:.2f}ms, P95 {p95_latency:.2f}ms")
                    else:
                        logger.warning(f"Could not find OpenVINO model.xml for benchmarking at {ov_model_xml}")
                except Exception as benchmark_err:
                    logger.warning(f"Failed to benchmark OpenVINO model latency: {benchmark_err}")

            if cm_path.exists():
                mlflow.log_artifact(str(cm_path), "production_models")
            if acc_path.exists():
                mlflow.log_artifact(str(acc_path), "production_models")

    logger.info("Anomaly detector retraining complete.")
    return True
