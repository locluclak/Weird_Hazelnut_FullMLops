import logging
import os
import subprocess
import sys

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from weird_hazelnut.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command, cwd=None, description=""):
    logger.info("Executing %s: %s", description, " ".join(command))
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Error executing %s:", description)
        logger.error(result.stderr)
        return False
    logger.info("Successfully executed %s", description)
    return True


def retrain_anomaly_detector(main_config, data_root):
    logger.info("--- Retraining Anomaly Detector ---")
    ad_project_dir = "../HazelnutAnomalyDetection"
    ad_config_path = os.path.join(ad_project_dir, "config/default.yaml")

    with open(ad_config_path, "r", encoding="utf-8") as f:
        ad_cfg = yaml.safe_load(f)

    ad_cfg["data"]["root"] = os.path.abspath(data_root)
    ad_cfg["mlflow"]["experiment_name"] = main_config["mlflow"]["experiment_name"]
    ad_cfg["mlflow"]["run_name"] = "Retraining_AD_" + os.path.basename(data_root)

    temp_config_path = "temp_ad_config.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(ad_cfg, f)

    try:
        return run_command(
            [sys.executable, "train.py", "--config", os.path.abspath(temp_config_path)],
            cwd=ad_project_dir,
            description="AD Training",
        )
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def retrain_classifier(main_config, data_root):
    logger.info("--- Retraining Classifier ---")
    cls_project_dir = "../HazelnutClassifier"
    cls_config_path = os.path.join(cls_project_dir, "config/default.yaml")

    return run_command(
        [
            sys.executable,
            "src/train.py",
            "--config",
            os.path.abspath(cls_config_path),
            "--data_dir",
            os.path.abspath(data_root),
            "--epochs",
            "5",
        ],
        cwd=cls_project_dir,
        description="Classifier Training",
    )


def register_latest_models(experiment_name):
    logger.info("--- Registering Models for experiment: %s ---", experiment_name)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.error("Experiment %s not found.", experiment_name)
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=10,
        )

        ad_run = None
        cls_run = None
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            if "Retraining_AD" in run_name and not ad_run:
                ad_run = run
            if "Retraining_Classifier" in run_name and not cls_run:
                cls_run = run
            if ad_run and cls_run:
                break

        if ad_run:
            mlflow.register_model(
                f"runs:/{ad_run.info.run_id}/production_models",
                "Hazelnut_Anomaly_Detector",
            )

        if cls_run:
            mlflow.register_model(
                f"runs:/{cls_run.info.run_id}/production_models",
                "Hazelnut_Classifier",
            )
    except Exception as e:
        logger.error("Error during model registration: %s", e)


def main():
    config = load_config()
    data_root = "data/final"

    if not os.path.exists(data_root):
        logger.error("Data root %s does not exist. Run sync_data.py first.", data_root)

    ad_success = retrain_anomaly_detector(config, data_root)
    cls_success = retrain_classifier(config, data_root)

    if ad_success and cls_success:
        logger.info("Retraining pipeline completed successfully for both models.")
        register_latest_models(config["mlflow"]["experiment_name"])
    else:
        logger.warning("Retraining pipeline finished with some errors. Skipping registration.")


if __name__ == "__main__":
    main()

