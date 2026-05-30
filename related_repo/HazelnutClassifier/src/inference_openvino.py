import argparse
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

from inference import HazelnutInference

class HazelnutOpenVINO:
    def __init__(self, xml_path: str, meta_path: str = None):
        self.xml_path = Path(xml_path)
        if meta_path is None:
            # Try to find meta in same dir as xml or parent
            meta_path = self.xml_path.parent / "training_meta.json"
            if not meta_path.exists():
                meta_path = self.xml_path.parent.parent / "training_meta.json"
        else:
            meta_path = Path(meta_path)
            
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
            
        self.classes = self.meta["classes"]
        self.img_size = self.meta["img_size"]
        
        # OpenVINO Core
        self.core = ov.Core()
        self.model = self.core.read_model(model=str(self.xml_path))
        
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_PRECISION_HINT": "f32",
        }
        self.compiled_model = self.core.compile_model(self.model, device_name="AUTO", config=config)
        self.output_layer = self.compiled_model.output(0)

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
        
        results_ov = self.compiled_model([img_data])[self.output_layer]
        logits = results_ov[0]
        
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

def safe_str(val):
    import sys
    try:
        s = str(val)
        enc = sys.stdout.encoding or 'utf-8'
        s.encode(enc)
        return s
    except Exception:
        return str(val).encode('ascii', errors='replace').decode('ascii')

def convert(onnx_path, output_dir):
    import openvino.tools.ovc as ovc
    import tempfile
    import shutil
    import gc
    
    print(f"Converting {safe_str(onnx_path)} to OpenVINO IR...")
    
    onnx_path = Path(onnx_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory with ASCII-only path to avoid Windows Unicode decoding issues in OpenVINO C++ frontend
    temp_dir = tempfile.TemporaryDirectory()
    try:
        temp_dir_path = Path(temp_dir.name)
        temp_onnx = temp_dir_path / onnx_path.name
        
        # Copy ONNX to temp ASCII path using its original name
        shutil.copy(str(onnx_path), str(temp_onnx))
        
        # Copy accompanying external data weight file if it exists
        external_data_src = onnx_path.parent / (onnx_path.name + ".data")
        if external_data_src.exists():
            external_data_dst = temp_dir_path / (onnx_path.name + ".data")
            shutil.copy(str(external_data_src), str(external_data_dst))
        
        # Convert model using temp paths
        ov_model = ovc.convert_model(str(temp_onnx))
        
        temp_xml = temp_dir_path / "hazelnut_classifier.xml"
        ov.save_model(ov_model, str(temp_xml))
        
        # Copy compiled files back to the target unicode folder
        shutil.copy(str(temp_xml), str(output_dir / "hazelnut_classifier.xml"))
        shutil.copy(str(temp_dir_path / "hazelnut_classifier.bin"), str(output_dir / "hazelnut_classifier.bin"))
        
        # Explicitly delete OpenVINO objects to release file handles
        del ov_model
        
    finally:
        # Run garbage collection to release any remaining C++ level references
        gc.collect()
        try:
            temp_dir.cleanup()
        except Exception as e:
            # Safely ignore cleanup errors on Windows as OS will handle it later
            print(f"[NOTE] Temporary directory cleanup deferred: {e}")
            
    print(f"OpenVINO IR successfully saved to: {safe_str(output_dir)}")

def benchmark(xml_path, onnx_path, meta_path, n_runs=100):
    print(f"Benchmarking OpenVINO vs ONNX ({n_runs} runs)...")
    ov_engine = HazelnutOpenVINO(xml_path, meta_path)
    onnx_engine = HazelnutInference(onnx_path, meta_path)
    
    # Dummy input
    dummy_img = np.random.randint(0, 255, (ov_engine.img_size, ov_engine.img_size, 3), dtype=np.uint8)
    img = Image.fromarray(dummy_img)
    img_data = ov_engine.preprocess(img)
    
    # Warmup
    for _ in range(10):
        ov_engine.compiled_model([img_data])
        onnx_engine.session.run(None, {onnx_engine.input_name: img_data})
        
    # ONNX Benchmark
    onnx_lats = []
    for _ in range(n_runs):
        s = time.perf_counter()
        onnx_engine.session.run(None, {onnx_engine.input_name: img_data})
        onnx_lats.append((time.perf_counter() - s) * 1000)
        
    # OpenVINO Benchmark
    ov_lats = []
    for _ in range(n_runs):
        s = time.perf_counter()
        ov_engine.compiled_model([img_data])
        ov_lats.append((time.perf_counter() - s) * 1000)
        
    print(f"ONNX Runtime:  Mean {np.mean(onnx_lats):.2f} ms, Median {np.median(onnx_lats):.2f} ms")
    print(f"OpenVINO:      Mean {np.mean(ov_lats):.2f} ms, Median {np.median(ov_lats):.2f} ms")
    print(f"Speedup:       {np.mean(onnx_lats) / np.mean(ov_lats):.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hazelnut OpenVINO Utilities")
    subparsers = parser.add_subparsers(dest="command")
    
    p_conv = subparsers.add_parser("convert")
    p_conv.add_argument("--onnx", type=str, required=True)
    p_conv.add_argument("--out", type=str, default="output/openvino")
    
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--model", type=str, required=True, help="Path to .xml")
    p_pred.add_argument("--image", type=str, required=True)
    p_pred.add_argument("--meta", type=str, help="Path to training_meta.json")
    
    p_bench = subparsers.add_parser("benchmark")
    p_bench.add_argument("--model", type=str, required=True, help="Path to .xml")
    p_bench.add_argument("--onnx", type=str, required=True)
    p_bench.add_argument("--meta", type=str, required=True)
    p_bench.add_argument("--n_runs", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.command == "convert":
        convert(args.onnx, args.out)
    elif args.command == "predict":
        engine = HazelnutOpenVINO(args.model, args.meta)
        res = engine.predict(args.image)
        print(json.dumps(res, indent=4))
    elif args.command == "benchmark":
        benchmark(args.model, args.onnx, args.meta, args.n_runs)
    else:
        parser.print_help()
