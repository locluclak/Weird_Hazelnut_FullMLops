from __future__ import annotations

import logging
from pathlib import Path

import mlflow

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

    logger.info("Anomaly detector retraining complete.")
    return True
