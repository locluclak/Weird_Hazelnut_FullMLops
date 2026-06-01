from __future__ import annotations

from pathlib import Path

from weird_hazelnut.config import load_config
from weird_hazelnut.evaluation.pipeline_eval import evaluate


def find_latest_file(root: str | Path, pattern: str) -> str:
    root = Path(root)
    candidates = list(root.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No file found: {root}/**/{pattern}")

    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main() -> None:
    config = load_config()

    ad_cfg = config.get("training", {}).get("anomaly_detector", {})
    cls_cfg = config.get("training", {}).get("classifier", {})

    ad_export_root = ad_cfg.get(
        "export_root",
        "models/anomaly_detector_retrained",
    )

    cls_output_dir = Path(
        cls_cfg.get(
            "output_dir",
            "models/classifier_retrained",
        )
    )

    candidate_paths = {
        "anomaly_model_path": find_latest_file(ad_export_root, "model.xml"),
        "classifier_model_path": str(
            cls_output_dir / cls_cfg.get("onnx_name", "model.onnx")
        ),
        "classifier_meta_path": str(
            cls_output_dir / cls_cfg.get("meta_name", "meta.json")
        ),
    }

    print("Evaluating candidate model paths:")
    for key, value in candidate_paths.items():
        print(f"  {key}: {value}")

    metrics = evaluate(
        config=config,
        model_overrides=candidate_paths,
        run_name="Candidate_Model_Evaluation",
    )

    if not metrics:
        raise RuntimeError("Candidate evaluation failed or returned no metrics.")

    print("Candidate evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()