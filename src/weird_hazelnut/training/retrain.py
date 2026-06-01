import logging

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_latest_models(experiment_name: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    logger.info("--- Registering Models for experiment: %s ---", experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.error("Experiment %s not found.", experiment_name)
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=25,
    )

    ad_run = None
    cls_run = None
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        if "Retraining_AD" in run_name and ad_run is None:
            ad_run = run
        if "Retraining_Classifier" in run_name and cls_run is None:
            cls_run = run
        if ad_run and cls_run:
            break

    if ad_run:
        mlflow.register_model(
            f"runs:/{ad_run.info.run_id}/production_models",
            "Hazelnut_Anomaly_Detector",
        )
    else:
        logger.warning("No anomaly detector retraining run found to register.")

    if cls_run:
        mlflow.register_model(
            f"runs:/{cls_run.info.run_id}/production_models",
            "Hazelnut_Classifier",
        )
    else:
        logger.warning("No classifier retraining run found to register.")

def run_retraining(config: dict, data_layer) -> dict:
    ad_enabled = config.get("training", {}).get("anomaly_detector", {}).get("enabled", True)
    cls_enabled = config.get("training", {}).get("classifier", {}).get("enabled", True)

    ad_success = True
    cls_success = True

    if ad_enabled:
        logger.info("--- Retraining Anomaly Detector ---")
        from weird_hazelnut.training.anomaly import train_anomaly_detector
        ad_success = train_anomaly_detector(config, data_layer)

    if cls_enabled:
        logger.info("--- Retraining Classifier ---")
        from weird_hazelnut.training.classifier import train_classifier
        cls_success = train_classifier(config, data_layer)

    return {
        "ad_enabled": ad_enabled,
        "classifier_enabled": cls_enabled,
        "ad_success": ad_success,
        "classifier_success": cls_success,
        "success": ad_success and cls_success,
    }


def main() -> None:
    config = load_config()
    data_layer = create_data_layer(config)
    if not data_layer:
        logger.error("Retraining now requires data_layer.enabled=true with MinIO/Postgres dataset items.")
        return

    train_count = len(data_layer.list_dataset_images(split="train"))
    test_count = len(data_layer.list_dataset_images(split="test"))
    logger.info("Using MinIO/Postgres dataset: %s train / %s test samples.", train_count, test_count)
    if train_count == 0 or test_count == 0:
        logger.error("No train/test dataset items found. Run scripts/migrate_dataset_to_datalake.py first.")
        return

    result = run_retraining(config, data_layer)

    if result["success"]:
        logger.info("Retraining pipeline completed successfully.")
        if config.get("registry", {}).get("enabled", False):
            register_latest_models(config["mlflow"]["experiment_name"])
    else:
        logger.warning("Retraining pipeline finished with errors. Skipping registration.")


if __name__ == "__main__":
    main()
