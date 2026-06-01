import argparse
import sys
from pathlib import Path
from PIL import Image

# Dynamic path resolution to find the 'src' directory from any folder level
current_dir = Path(__file__).resolve().parent
src_dir = None
for _ in range(4):
    if (current_dir / "src").is_dir():
        src_dir = current_dir / "src"
        break
    current_dir = current_dir.parent

if src_dir:
    sys.path.append(str(src_dir))
else:
    sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from weird_hazelnut.config import load_config
from weird_hazelnut.inference import HazelnutClassifier

def main():
    parser = argparse.ArgumentParser(description="Test Classifier model on a single image.")
    parser.add_argument("image_path", type=str, help="Path to the image to classify.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models/classifier_retrained", 
        help="Directory containing the model.onnx and meta.json. (Defaults to models/classifier_retrained)"
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "model.onnx"
    meta_path = model_dir / "meta.json"

    if not onnx_path.exists() or not meta_path.exists():
        # Fallback to config path if the retrained folder does not exist
        print(f"Warning: Retrained model not found in '{model_dir}'. Falling back to config paths...")
        config = load_config()
        cls_cfg = config.get("models", {}).get("classifier", {})
        onnx_path = Path(cls_cfg.get("model_path", "models/classifier/model.onnx"))
        meta_path = Path(cls_cfg.get("meta_path", "models/classifier/meta.json"))

    if not onnx_path.exists() or not meta_path.exists():
        print(f"Error: ONNX model or meta.json not found at '{onnx_path}' or '{meta_path}'.")
        sys.exit(1)

    print(f"--- Loading Classifier ---")
    print(f"ONNX Model: {onnx_path}")
    print(f"Meta JSON : {meta_path}")
    
    try:
        classifier = HazelnutClassifier(str(onnx_path), str(meta_path))
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        sys.exit(1)

    print(f"\n--- Running Inference ---")
    print(f"Target Image: {image_path}")
    
    try:
        image = Image.open(image_path)
        result = classifier.predict(image)
        
        print("\n--- Result ---")
        print(f"Predicted Class: {result['label'].upper()}")
        print(f"Confidence     : {result['confidence'] * 100:.2f}%")
        print(f"Latency        : {result['latency_ms']:.2f} ms")
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
