# WeirdHazelnut

WeirdHazelnut is a hazelnut quality-control MLOps demo. It combines:

- OpenVINO anomaly detection
- ONNX defect classification
- FastAPI inference
- MinIO image/object storage
- Postgres metadata, labels, and dataset tracking
- Label Studio human review for uncertain images
- MLflow tracking and model registry

The normal startup path is Docker Compose.

## Project Layout

- `api.py` - starts the FastAPI inference service.
- `sync_data.py` - syncs completed Label Studio labels into Postgres.
- `retrain.py` - retrains/registers models through integrated training code.
- `evaluate_pipeline.py` - evaluates the pipeline from the MinIO/Postgres test split.
- `src/weird_hazelnut/` - main application package.
- `config.yaml` - local config.
- `config.docker.yaml` - Docker config.
- `ARCHITECTURE.md` - architecture notes and AI working guide.

Large runtime folders:

- `data/` - seed datasets and manual test files only.
- `models/` - required model artifacts.
- `mlruns/`, `mlflow.db` - MLflow runtime state.
- `openvino_cache/` - generated OpenVINO cache.

## Requirements

- Docker Desktop
- Docker Compose v2
- Python 3.10 if running locally

Required model files:

- `models/anomaly_detector/model.xml`
- `models/anomaly_detector/model.bin`
- `models/classifier/model.onnx`
- `models/classifier/meta.json`

## Start With Docker

Build and start all services:

```bash
docker compose up -d --build
```

Services:

- API: `http://localhost:8000`
- Label Studio: `http://localhost:8080`
- MLflow: `http://localhost:5000`
- MinIO API: `http://localhost:9000`
- MinIO console: `http://localhost:9001`
- Postgres: `localhost:5432`

Check containers:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f app
```

## Initialize Label Studio

After the containers are running:

1. Open `http://localhost:8080`.
2. Login with the credentials below.
3. Create or copy your Label Studio API key.
4. Paste it into `label_studio.api_key` in `config.docker.yaml`.
5. Restart the API container so it reloads the updated config:

```bash
docker compose restart app
```

Then create the Label Studio project:

```bash
docker compose exec app python labeling/init_project.py
```

The init script updates `label_studio.project_id` in `config.docker.yaml`.
Tasks use MinIO presigned URLs; Label Studio local file storage is not configured.
Restart the API container one more time after project creation:

```bash
docker compose restart app
```

Login:

- URL: `http://localhost:8080`
- Email: `admin@example.com`
- Password: `admin123456`

Docker uses `config.docker.yaml`, not `config.yaml`. The compose file mounts `config.docker.yaml` into the app container, so you do not need to rebuild after editing the API key, but the running API process still needs a restart to reload it.

## Initialize The Data Lake

On first setup, seed MinIO and Postgres from the local `data/hazelnut` folder:

```bash
docker compose exec app python scripts/migrate_dataset_to_datalake.py --dataset-root data/hazelnut --dataset-name initial_hazelnut
```

After migration, MinIO/Postgres is the dataset source of truth. Keep `/data` for
initial seeding and manual API tests only.

## Test The API

Health check:

```bash
curl http://localhost:8000/health
```

Run prediction:

```bash
curl.exe -X POST http://localhost:8000/predict -F "file=@data/hazelnut/test/good/000.png"
```

Example response:

```json
{
  "anomaly": true,
  "anomaly_score": 0.81,
  "uncertain": false,
  "ad_latency_ms": 35.0,
  "model_b_called": true,
  "classifier_latency_ms": 9.0,
  "label": "crack",
  "confidence": 0.92,
  "total_latency_ms": 45.0,
  "ls_task_id": null
}
```

Routing behavior:

- `score < 0.3` records the image as `normal`.
- `0.3 <= score <= 0.7` records the image as `uncertain` and creates a Label Studio task.
- `score > 0.7` runs the classifier and records the image as `anomalies`.

When `data_layer.enabled` is true, the API uploads each image to MinIO,
deduplicates it by SHA256 in Postgres, records each inference run, and tracks
Label Studio task IDs. Label Studio receives MinIO presigned URLs and does not
need a local copy under `data/lake`.

Thresholds are configured in `config.docker.yaml` for Docker and `config.yaml` for local runs.

## Human Labeling Loop

1. Open `http://localhost:8080`.
2. Label uncertain images as `Normal` or `Anomaly`.
3. If `Anomaly`, choose the defect class: `crack`, `cut`, `hole`, or `print`.
4. Submit the annotation.

Sync completed labels:

```bash
docker compose exec app python sync_data.py
```

With the data layer enabled, sync writes completed labels into the `annotations`
table. It does not copy images into `data/final`.

## Evaluate

Run evaluation against the MinIO/Postgres `test` split:

```bash
docker compose exec app python evaluate_pipeline.py
```

Results are logged to MLflow.

## Retrain

Run:

```bash
docker compose exec app python retrain.py
```

Retraining is now self-contained in `src/weird_hazelnut/training/` and no longer
requires `HazelnutAnomalyDetection` or `HazelnutClassifier` beside this repo.
The classifier trainer streams samples from MinIO/Postgres. The anomaly trainer
also reads the canonical dataset from MinIO/Postgres, then uses a temporary
system directory as an Anomalib `Folder` bridge because Anomalib's training API
is path-based; it does not write training data into `/data` or `data/final`.

Training parameters live under `training` in `config.yaml` and
`config.docker.yaml`. Both training runs log parameters, dataset counts,
metrics, and production artifacts to MLflow. If `registry.enabled` is true,
`retrain.py` registers the latest `Retraining_AD_*` and
`Retraining_Classifier_*` runs as `Hazelnut_Anomaly_Detector` and
`Hazelnut_Classifier`.

The initial MVTec-style seed usually has only `train/good`, which is enough for
anomaly retraining. Classifier retraining requires defect-labeled training items
such as `crack`, `cut`, `hole`, and `print` in Postgres.

## Local Development

Create an environment and install dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Run API locally:

```bash
python api.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

Run a manual pipeline smoke test:

```bash
python test_pipeline.py
```

## Useful Commands

Stop services:

```bash
docker compose down
```

Restart services:

```bash
docker compose restart
```

Rebuild after dependency or Dockerfile changes:

```bash
docker compose up -d --build
```

Open MLflow logs:

```bash
docker compose logs -f mlflow
```

Open Label Studio logs:

```bash
docker compose logs -f label-studio
```

## Troubleshooting

If Label Studio initialization fails with a service-name/DNS error:

```bash
docker compose ps
docker compose logs label-studio
```

Then retry:

```bash
docker compose exec app python labeling/init_project.py
```

If Label Studio fails with `Data directory is not writable: /label-studio/data`, recreate its Docker-managed volume:

```bash
docker compose down
docker volume rm weirdhazelnut_label_studio_data
docker compose up -d label-studio
```

The compose file intentionally uses a named volume for `/label-studio/data`. Label Studio runs as non-root UID `1001`, and host bind mounts can fail when the mounted directory is owned by `root` without root-group write permission.

If imports fail locally, install dependencies from `requirements.txt`. The project depends on heavy ML packages including `mlflow`, `anomalib`, `onnxruntime`, `openvino`, and `label-studio-sdk`.

If model loading fails, verify the files under `models/` exist and match the config paths.

If the API fails during startup with a Postgres or MinIO connection error, check:

```bash
docker compose ps
docker compose logs postgres
docker compose logs minio
```

The app bootstraps the database schema and MinIO bucket on startup when
`data_layer.enabled` is true.

If uncertain images do not appear in Label Studio, check:

- `label_studio.project_id` in the active config
- `label_studio.api_key`
- `data_layer.object_storage.public_endpoint` points to a browser-reachable MinIO endpoint
- MinIO is reachable at `http://localhost:9000`

If a task opens but the image fails with `There was an issue loading URL from $image value`, verify the task image URL is a MinIO presigned URL using the configured public endpoint, for example:

```text
http://localhost:9000/weird-hazelnut/raw/...
```

Tasks created before the MinIO URL change can still contain old local-file URLs; delete or recreate those tasks after restarting Label Studio and the app.

The init script is safe to rerun:

```bash
docker compose exec app python labeling/init_project.py
```

It is safe to rerun. If `label_studio.project_id` already exists, it reuses that project and only creates the missing storage.

If Label Studio was reset or its Docker volume was recreated, `config.docker.yaml` can still contain a stale `label_studio.project_id`. The init script detects that case, creates a new project, and rewrites `project_id`. After it changes `project_id`, restart the app:

```bash
docker compose restart app
```

## Notes For Maintainers

- Read `ARCHITECTURE.md` before changing the system.
- Do not commit runtime artifacts from `data/`, `models/`, `mlruns/`, `mlflow.db`, or `openvino_cache/`.
- Root scripts are compatibility wrappers; put new implementation code under `src/weird_hazelnut/`.
