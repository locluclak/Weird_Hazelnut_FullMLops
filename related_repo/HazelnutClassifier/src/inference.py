import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

class HazelnutInference:
    def __init__(self, onnx_path: str, meta_path: str = None):
        self.onnx_path = Path(onnx_path)
        if meta_path is None:
            meta_path = self.onnx_path.parent / "training_meta.json"
        else:
            meta_path = Path(meta_path)
            
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
            
        self.classes = self.meta["classes"]
        self.img_size = self.meta["img_size"]
        
        # ONNX Runtime Session Config
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        
        self.session = ort.InferenceSession(
            str(self.onnx_path), 
            sess_options=opts, 
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: Image.Image):
        # Match training preprocessing
        image = image.convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_data = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        # HWC -> CHW
        img_data = img_data.transpose(2, 0, 1)
        # Add batch dim
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
        return img_data

    def predict(self, image_path: str, top_k: int = 5):
        image = Image.open(image_path)
        return self.predict_pil(image, top_k)

    def predict_pil(self, image: Image.Image, top_k: int = 5):
        start_time = time.perf_counter()
        img_data = self.preprocess(image)
        
        outputs = self.session.run(None, {self.input_name: img_data})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        idx = np.argsort(probs)[::-1]
        
        results = []
        for i in range(min(top_k, len(self.classes))):
            results.append({
                "class": self.classes[idx[i]],
                "prob": float(probs[idx[i]])
            })
            
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "class": results[0]["class"],
            "confidence": results[0]["prob"],
            "top_k": results,
            "latency_ms": latency
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hazelnut ONNX Inference")
    parser.add_argument("onnx_path", type=str, help="Path to ONNX model")
    parser.add_argument("meta_path", type=str, nargs="?", help="Path to training_meta.json")
    parser.add_argument("image_path", type=str, nargs="?", help="Path to image to predict")
    
    args = parser.parse_args()
    
    engine = HazelnutInference(args.onnx_path, args.meta_path)
    
    if args.image_path:
        res = engine.predict(args.image_path)
        print(json.dumps(res, indent=4))
    else:
        print(f"Model loaded. Classes: {engine.classes}")
