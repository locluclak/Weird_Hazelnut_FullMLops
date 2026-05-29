from src.weird_hazelnut.training.retrain import (
    main,
    register_latest_models,
    retrain_anomaly_detector,
    retrain_classifier,
    run_command,
)

__all__ = [
    "main",
    "register_latest_models",
    "retrain_anomaly_detector",
    "retrain_classifier",
    "run_command",
]


if __name__ == "__main__":
    main()
