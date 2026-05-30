import base64
import io
import json
import time
import os
import argparse
import sys
from contextlib import asynccontextmanager
from typing import List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Add current folder to sys.path for robust modular imports
sys.path.append(str(Path(__file__).parent))

import mlflow
from inference import HazelnutInference

# Schemas
class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    top_k: List[dict]
    latency_ms: float

class Base64Request(BaseModel):
    image_base64: str
    top_k: int = 5

# Global state
engine = None
start_time = time.time()
mlflow_run = None
request_count = 0
request_metrics = {
    "confidence": [],
    "latency_ms": [],
    "pred_label": []
}

def _log_mlflow_figure(name: str, fig) -> None:
    """Log a matplotlib figure to MLflow if the run is active."""
    if not mlflow_run:
        return
    try:
        mlflow.log_figure(fig, name)
    except Exception as e:
        print(f"[WARNING] Failed to log MLflow artifact '{name}': {e}")
    finally:
        try:
            fig.clf()
        except Exception:
            pass

def log_mlflow_histograms(force: bool = False) -> None:
    """Create and log histogram artifacts for collected request metrics."""
    if not mlflow_run or len(request_metrics["confidence"]) == 0:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')  # Headless mode to prevent GUI thread issues
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARNING] matplotlib import failed; histogram artifacts will not be generated: {e}")
        return

    try:
        # 1. Prediction Confidence distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(request_metrics["confidence"], bins=20, range=(0, 1), color="#1f77b4", edgecolor="black")
        ax.set_title("Prediction Confidence Distribution")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        _log_mlflow_figure("histogram_confidence.png", fig)
        plt.close(fig)

        # 2. Latency distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(request_metrics["latency_ms"], bins=20, color="#2ca02c", edgecolor="black")
        ax.set_title("Inference Latency Distribution")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        _log_mlflow_figure("histogram_latency_ms.png", fig)
        plt.close(fig)

        # 3. Label/Class distribution (Tailored Multi-Class Bar Plot)
        labels = request_metrics["pred_label"]
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        # Ensure all classes are represented even if count is 0
        full_counts = {c: 0 for c in engine.classes}
        for ul, lc in zip(unique_labels, label_counts):
            full_counts[ul] = lc
            
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(list(full_counts.keys()), list(full_counts.values()), color="#c44e52", edgecolor="black")
        ax.set_title("Prediction Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30)
        fig.tight_layout()
        _log_mlflow_figure("histogram_pred_label.png", fig)
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] Failed to generate MLflow histogram artifacts: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, mlflow_run
    
    # 1. Load centralized configuration
    config_path = Path("config/default.yaml")
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
        
    mlflow_cfg = cfg.get("mlflow", {})
    export_cfg = cfg.get("export", {})
    
    engine_type = os.getenv("ENGINE", "onnx").lower()
    onnx_name = export_cfg.get("onnx_name", "hazelnut_classifier.onnx")
    onnx_path = os.getenv("ONNX_PATH", f"output/{onnx_name}")
    
    # Resolve OpenVINO paths
    ov_dir = export_cfg.get("openvino_dir", "output/openvino")
    ov_xml_path = os.getenv("OV_XML_PATH", f"{ov_dir}/hazelnut_classifier.xml")
    meta_path = os.getenv("META_PATH", "output/training_meta.json")
    
    # Fallback checks if paths do not exist in default directory
    if engine_type == "onnx" and not Path(onnx_path).exists():
        if Path(onnx_name).exists():
            onnx_path = onnx_name
        elif Path(f"output/{onnx_name}").exists():
            onnx_path = f"output/{onnx_name}"
            
    try:
        if engine_type == "openvino":
            from inference_openvino import HazelnutOpenVINO
            print(f"Loading OpenVINO engine from {ov_xml_path}...")
            engine = HazelnutOpenVINO(ov_xml_path, meta_path)
        else:
            print(f"Loading ONNX engine from {onnx_path}...")
            engine = HazelnutInference(onnx_path, meta_path)
            
        print(f"Model loaded successfully. Classes: {engine.classes}")
    except Exception as e:
        print(f"ERROR: Could not load model. API will fail on predict. Details: {e}")
        
    # 2. Initialize MLflow Online Tracking with Resilient Fallback
    try:
        tracking_uri = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")
        if tracking_uri.startswith("http"):
            import urllib.request
            try:
                # 2-second timeout connection ping
                urllib.request.urlopen(tracking_uri, timeout=2.0)
                print(f"[MLflow] Connected successfully to tracking server at: {tracking_uri}")
            except Exception:
                print(f"[MLflow] [WARNING] Tracking server at {tracking_uri} is offline or unreachable.")
                print("[MLflow] Falling back to local file-based tracking at 'file:./mlruns' to prevent API crash.")
                tracking_uri = "file:./mlruns"
                
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_cfg.get("api_experiment_name", "hazelnut_classifier_production_api"))
        
        # Track live server system hardware resources (CPU, RAM, Network...)
        mlflow.enable_system_metrics_logging()
        
        mlflow_run = mlflow.start_run(run_name=f"api_session_{int(time.time())}")
        mlflow.log_params({
            "serving_engine": engine_type,
            "onnx_path": onnx_path,
            "ov_xml_path": ov_xml_path,
            "meta_path": meta_path,
            "img_size": engine.img_size if engine else 0
        })
        print(f"MLflow live tracking successfully initialized.")
    except Exception as mlflow_err:
        print(f"[WARNING] MLflow online tracking setup failed: {mlflow_err}. API will run without MLOps telemetry logging.")
        
    yield
    
    # Cleanup on Server Shutdown
    if mlflow_run:
        try:
            log_mlflow_histograms()
            mlflow.end_run()
            print("Successfully saved final telemetry distributions and closed MLflow run.")
        except Exception as shutdown_err:
            print(f"[WARNING] MLflow shutdown cleanups failed: {shutdown_err}")

app = FastAPI(
    title="Hazelnut Quality Classifier REST Service",
    description="High-performance classification engine optimized for CPU with live MLflow telemetry metrics.",
    version="1.2.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_s": time.time() - start_time,
        "engine": engine.__class__.__name__ if engine else None,
        "classes": engine.classes if engine else [],
        "img_size": engine.img_size if engine else 0,
        "total_requests": request_count
    }

def process_image(img_bytes):
    global request_count
    try:
        img = Image.open(io.BytesIO(img_bytes))
        res = engine.predict_pil(img)
        
        request_count += 1
        conf = float(res["confidence"])
        latency = float(res["latency_ms"])
        label = res["class"]
        
        request_metrics["confidence"].append(conf)
        request_metrics["latency_ms"].append(latency)
        request_metrics["pred_label"].append(label)
        
        # Log to active MLflow serving run
        if mlflow_run:
            try:
                # Map label to numerical ID for plotting in step metrics
                label_id = engine.classes.index(label) if label in engine.classes else -1
                mlflow.log_metrics({
                    "inference_confidence": conf,
                    "inference_label_id": float(label_id),
                    "latency_ms": latency,
                    "total_requests": float(request_count)
                }, step=request_count)
                
                # Refresh dynamic histograms
                log_mlflow_histograms(force=True)
            except Exception as mlflow_log_err:
                print(f"[WARNING] MLflow telemetry request log failed: {mlflow_log_err}")
                
        return {
            "class_name": label,
            "confidence": conf,
            "top_k": res["top_k"],
            "latency_ms": latency
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or inference error: {e}")

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if engine is None: 
        raise HTTPException(status_code=503, detail="Model classifier not loaded or failed to initialize.")
    content = await file.read()
    return process_image(content)

@app.post("/predict/base64", response_model=PredictionResult)
async def predict_base64(req: Base64Request):
    if engine is None: 
        raise HTTPException(status_code=503, detail="Model classifier not loaded or failed to initialize.")
    try:
        img_bytes = base64.b64decode(req.image_base64)
        return process_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding or image format: {e}")

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if engine is None: 
        raise HTTPException(status_code=503, detail="Model classifier not loaded or failed to initialize.")
    if len(files) > 64: 
        raise HTTPException(status_code=400, detail="Batch execution limited to a maximum of 64 files per query.")
    
    results = []
    for file in files:
        content = await file.read()
        results.append(process_image(content))
        
    return {
        "count": len(results),
        "predictions": results
    }

@app.get("/benchmark")
async def benchmark(n_runs: int = 50):
    if engine is None: 
        raise HTTPException(status_code=503, detail="Model classifier not loaded or failed to initialize.")
    
    dummy_img = np.random.randint(0, 255, (engine.img_size, engine.img_size, 3), dtype=np.uint8)
    img = Image.fromarray(dummy_img)
    
    latencies = []
    for _ in range(n_runs):
        res = engine.predict_pil(img)
        latencies.append(res["latency_ms"])
        
    return {
        "n_runs": n_runs,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "engine": engine.__class__.__name__
    }

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine", choices=["onnx", "openvino"], default="onnx")
    args = parser.parse_args()
    
    os.environ["ENGINE"] = args.engine
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
