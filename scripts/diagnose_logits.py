import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

# Add src/ to python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer
from weird_hazelnut.evaluation.pipeline_eval import _iter_test_images

def main():
    config = load_config()
    data_layer = create_data_layer(config)
    
    onnx_path = config["models"]["classifier"]["model_path"]
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    input_name = sess.get_inputs()[0].name
    img_size = 224
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    classes = ["crack", "cut", "hole", "print"]
    
    print(f"Evaluating ONNX Model Raw Output Logits (Model: {onnx_path}):")
    count = 0
    for image, true_label in _iter_test_images(data_layer):
        if true_label == "good":
            continue
            
        # Preprocess image
        image = image.convert("RGB")
        image = image.resize((img_size, img_size), Image.BILINEAR)
        img_data = np.array(image).astype(np.float32) / 255.0
        img_data = (img_data - mean) / std
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
        
        # Run inference
        logits = sess.run(None, {input_name: img_data})[0][0]
        max_idx = np.argmax(logits)
        predicted_label = classes[max_idx]
        
        print(f"  Sample {count+1:02d} | True: {true_label:<6} | Pred: {predicted_label:<6} | Logits: {list(np.round(logits, 4))}")
        
        count += 1
        if count >= 30:
            break

if __name__ == "__main__":
    main()
