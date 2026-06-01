import sys
from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer


LABELS = ["good", "crack", "cut", "hole", "print"]
DEFECT_LABELS = ["crack", "cut", "hole", "print"]


def main():
    config = load_config()
    data_layer = create_data_layer(config)

    if not data_layer:
        print("DATA_GATE_FAILED: data_layer.enabled must be true")
        sys.exit(1)

    counts = {}

    for split in ["train", "val", "test"]:
        for label in LABELS:
            counts[f"{split}/{label}"] = len(
                data_layer.list_dataset_images(split=split, labels=[label])
            )

    counts["train/total"] = sum(counts[f"train/{label}"] for label in LABELS)
    counts["train/defect_total"] = sum(
        counts[f"train/{label}"] for label in DEFECT_LABELS
    )
    counts["val/total"] = sum(counts[f"val/{label}"] for label in LABELS)
    counts["test/total"] = sum(counts[f"test/{label}"] for label in LABELS)

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

    print("DATASET_COUNTS:")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    if reasons:
        print("DATA_GATE_FAILED:")
        for reason in reasons:
            print(f"  - {reason}")
        sys.exit(99)

    print("DATA_GATE_PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()