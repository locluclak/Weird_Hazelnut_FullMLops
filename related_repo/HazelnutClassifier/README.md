# Hazelnut Defect Classifier (Production-Grade MLOps Pipeline)

This project implements a lightweight hazelnut quality inspection system using PyTorch, MobileNetV3-Large, and optimized ONNX/OpenVINO inference. 

It has been upgraded to implement **professional production-ready MLOps workflows** including centralized YAML configurations, automated data leakage checker, full MLflow experiment tracking (for both training and validation runs), and live dynamic Matplotlib telemetry serving on FastAPI.

---

## 🚀 Key Upgraded Features

- **Unsupervised Anomaly vs Supervised Defect Task**: 
  Unlike unsupervised anomaly segmentation (which trains only on normal `good` data), this project implements a **supervised multi-class classification** system (classes: `good`, `cut`, `hole`, `print`, `crack`) using `MobileNetV3-Large`. It does not require pixel overlays but is optimized for high-speed multi-class classification.
- **Centralized Configuration**: 
  All hyperparameters, dataset directories, and logging configurations are managed in one place under `config/default.yaml`.
- **Automated Data Leakage Checker**: 
  Automatically checks for duplicate image bhashes (MD5 verification) between `train` and `test`/`val` splits at training startup to ensure evaluation integrity.
- **MLflow Tracking**: 
  Training hyperparameters, dataset stats, and epoch loss/accuracy are automatically logged to a central MLflow tracking server.
- **MLflow Artifact Plots**: 
  Auto-generates high-DPI Confusion Matrices (5x5 multi-class), per-class accuracy bar charts, training histories, and test validation grids, uploading them directly to the active MLflow run.
- **REST API Serving & Real-Time Telemetry**: 
  The FastAPI service handles lifespan startup/shutdown runtime logs, dynamic hardware metric tracking (CPU, memory, networking) via `psutil`, and dynamically plots and uploads Matplotlib histograms (`histogram_confidence.png`, `histogram_latency.png`, `histogram_pred_label.png`) to MLflow live with each request.
- **CPU Optimized Inference**: 
  Achieves highly optimized CPU inference latency (~2ms) using ONNX Runtime and Intel OpenVINO.

---

## 🛠️ Setup

### 1. Environment Installation
Install all base dependencies along with MLOps libraries (MLflow, PyYAML, psutil):

```bash
conda activate adnut
pip install -r requirements.txt
```

### 2. Dataset Split & Check
Split the labeled Hazelnut test set into train/val/test folders and verify class balance:

```bash
# Split dataset 70% Train, 10% Val, 20% Test
python src/prepare_data.py full_split data/hazelnut/test data/final --train 0.7 --val 0.1 --test 0.2

# Check splits for imbalance or small count warnings
python src/prepare_data.py check data/final
```

---

## 📂 Project Structure

```text
WeirdHazelnut_classifier/
├── config/
│   └── default.yaml          # Centralized configuration YAML
├── src/
│   ├── api.py                # FastAPI REST Server with live MLflow telemetry
│   ├── train.py              # Supervised Training with progressive unfreezing & MLflow tracking
│   ├── prepare_data.py       # Data split, MVTec convert, and checking CLI
│   ├── data_utils.py         # Automated MD5 duplicate data leakage checker
│   ├── inference.py          # ONNX Runtime CPU inference runner
│   ├── inference_openvino.py # OpenVINO compilation, prediction, and benchmark CLI
│   └── evaluate.py           # Offline evaluation charts helper
├── output/                   # Trained models, ONNX exports, and evaluation charts
├── requirements.txt          # Upgraded project dependencies
└── README.md                 # Project Documentation
```

---

## 🏋️ Training & Experiment Tracking

Run the training pipeline with automated data leakage verification and full MLflow tracking:

```bash
python src/train.py --config config/default.yaml
```

- **MLflow Tracking UI**: Launch the MLflow UI by running `mlflow ui` in the background and navigate to `http://127.0.0.1:5000`.
- **Logged Parameters**: AdamW optimizer values, progressive unfreezing thresholds, mixup levels, data leakage flags, and counts per class.
- **Logged Metrics**: Step-by-step training and validation loss/accuracy mapped per epoch.
- **Logged Artifacts**: 5x5 Confusion Matrix heatmaps, per-class accuracy charts, loss/accuracy curves, a sample prediction grid, the exported `hazelnut_classifier.onnx` model, and `training_meta.json`.

---

## 🔍 Offline Evaluation (CLI)

Although the training script **automatically** runs evaluation and uploads all charts to MLflow at completion, you can still run manual offline evaluation for any exported ONNX model at any time using:

```bash
python src/evaluate.py --onnx output/hazelnut_classifier.onnx --meta output/training_meta.json --data_dir data/final --output_dir output/
```

**Key Generated Charts (Saved in `output/`):**
*   `confusion_matrix.png`: 5x5 multi-class confusion matrix with annotations.
*   `per_class_accuracy.png`: Accuracy per class (Green $\ge 90\%$, Orange $75-90\%$, Red $< 75\%$).
*   `training_history.png`: Visual loss and accuracy trends per epoch.
*   `sample_predictions.png`: Grid of test images with prediction vs ground truth (Green/Red color-coded borders).

---


## 🌐 API Serving & Real-Time Telemetry

Serve the defect classifier as a high-performance web service with real-time operational dashboard metrics.

### 1. Launch the Server
```bash
# Option A: Start using ONNX Runtime CPU engine (Default)
python src/api.py --engine onnx

# Option B: Start using optimized Intel OpenVINO engine
python src/api.py --engine openvino
```
The FastAPI server will start at `http://localhost:8000`.

### 2. Live Monitoring
Open `http://127.0.0.1:5000` to view the ongoing API serving run (`api_session_<timestamp>`):
- **Live System Metrics**: CPU utilization, RAM usage, disk I/O, and socket networking tracked in real-time.
- **Dynamic Telemetry Logs**: Confidence scores, latencies, and prediction classes logged step-by-step for every incoming request.
- **Live Distributions**: Matplotlib histogram figures of latencies, prediction labels, and confidences updated and uploaded dynamically!

### 3. API Usage Examples
Test the live endpoints using standard CLI commands:

- **Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Single Image Prediction**:
  ```bash
  curl.exe -X POST http://localhost:8000/predict -F "file=@data/final/test/crack/000.png"
  ```
- **Batch Image Prediction (Max 64)**:
  ```bash
  curl -X POST http://localhost:8000/predict/batch -F "files=@data/final/test/crack/000.png" -F "files=@data/final/test/good/000.png"
  ```
- **Speed Benchmarking**:
  ```bash
  curl "http://localhost:8000/benchmark?n_runs=50"
  ```

---

## ⚙️ Centralized Configuration
Parameters in `config/default.yaml` can be adjusted to customize:
- **Data**: Splits, augmentation levels (`light`, `medium`, `heavy`), input size.
- **Training**: Optimizer values, early stopping patience, mixup beta distribution.
- **MLflow**: Central tracking servers, experiment names, and run namespaces.
- **Exports**: ONNX names and OpenVINO weight compilation directories.

---

## 📋 Implementation Spec & Technical Architecture

For convenience, the noteworthy architectural choices and specifications defined in `hazelnut_classifier_spec.md` have been fully integrated here:

### 1. Model Architecture
*   **Backbone**: `MobileNetV3-Large` (or `Small` based on preferences) pretrained on ImageNet.
*   **Classifier Head**: Replaced with a custom dropout-regularized classification layer:
    ```python
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)  # 5 classes
    )
    ```

### 2. Progressive Unfreezing
To ensure stable transfer learning from ImageNet:
*   **Epochs 1 to `epochs // 3`**: Backbone feature extractors are **frozen** (except for classifier head and the final `features[13:]` block), training only the high-level semantic representations.
*   **Epochs `epochs // 3` onwards**: **All layers are unfrozen** and trainable. The optimizer is reset with a reduced learning rate (`lr * 0.1`) alongside a reset cosine scheduler to fine-tune the entire network.

### 3. Class Imbalance & Regularization
*   **Weighted Sampling**: Employs PyTorch `WeightedRandomSampler` during batch collection where each sample's sampling probability is proportional to $1 / \text{class\_count}$.
*   **Loss Weighting**: CrossEntropyLoss is configured with inverse class frequency weights to penalize minority class classification errors.
*   **Label Smoothing**: Configures `label_smoothing=0.1` to prevent overconfidence and improve out-of-distribution generalization.
*   **Mixup Augmentation**: When `mixup_alpha > 0`, mixes pair samples dynamically in each training batch using standard interpolation parameters $\lambda \sim \text{Beta}(0.2, 0.2)$ and calculates mixed cross-entropy loss criteria.

### 4. Advanced CPU Optimization & Inference Specs
Optimized to execute real-time industrial inference on Intel CPUs under ~2ms:
*   **PyTorch Threads**: Automatically limits intra/inter-op threads to prevent CPU over-subscription context switching on high-core servers.
*   **ONNX Runtime Settings**: Configured with `ORT_ENABLE_ALL` graph optimization levels, `CPUExecutionProvider`, and dynamic batch dimensions (`batch` axis on inputs/outputs).
*   **OpenVINO Optimization**: Converted ONNX models to OpenVINO IR (`.xml` / `.bin`) via Intel OVC. The Core compiler is set with the `LATENCY` performance hint and `NUM_STREAMS=1` to achieve maximum frame-rate throughput.

### 5. Production API Serving Schema
Exposes REST endpoints built on top of high-performance asynchronous FastAPI:
*   **`GET /health`**: Returns engine class, class labels, n_requests, target image sizes, and system uptime.
*   **`POST /predict`**: Accepts multipart raw image upload, decodes and runs inference with full telemetry logging.
*   **`POST /predict/base64`**: Accepts base64 encoded image strings to integrate with remote edge hardware clients.
*   **`POST /predict/batch`**: Handles array inputs of multiple files (up to a maximum of 64) for parallel processing.
*   **`GET /benchmark`**: Executes 50 test iterations dynamically to output performance statistics (Mean, Median, Std, Min, Max execution latencies).

