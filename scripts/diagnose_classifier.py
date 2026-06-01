import sys
from pathlib import Path
from collections import Counter

# Add src/ to python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer
from weird_hazelnut.pipeline import HazelnutPipeline
from weird_hazelnut.evaluation.pipeline_eval import _iter_test_images

def main():
    config = load_config()
    data_layer = create_data_layer(config)
    pipeline = HazelnutPipeline(
        anomaly_model_path=config["models"]["anomaly_detector"]["path"],
        classifier_model_path=config["models"]["classifier"]["model_path"],
        classifier_meta_path=config["models"]["classifier"]["meta_path"],
        threshold_low=config["pipeline"]["thresholds"]["low"],
        threshold_high=config["pipeline"]["thresholds"]["high"],
        lake_dir=config["pipeline"]["lake_dir"],
        data_layer=data_layer,
    )
    
    print("Classes in meta:", pipeline.classifier.classes)
    
    correct = 0
    total = 0
    predictions = []
    
    for image, true_label in _iter_test_images(data_layer):
        if true_label == "good":
            continue
        res = pipeline.run(image, persist=False)
        if res["anomaly"]:
            predictions.append((true_label, res["label"]))
            if res["label"] == true_label:
                correct += 1
            total += 1
            
    print(f"Total anomalies detected: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct/total if total > 0 else 0:.4f}")
    
    print("\nTrue labels count in predictions:", Counter([t for t, _ in predictions]))
    print("Predicted labels count:", Counter([p for _, p in predictions]))
    
    print("\nDetailed list of first 30 predictions (true_label -> predicted_label):")
    for idx, (t, p) in enumerate(predictions[:30]):
        status = "✅" if t == p else "❌"
        print(f"  {idx+1:02d}. {t} -> {p} {status}")

if __name__ == "__main__":
    main()
