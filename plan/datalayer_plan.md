# WeirdHazelnut Data Layer Integration Plan

## Goal
Introduce a real data layer for WeirdHazelnut using:

- `MinIO` for image/object storage
- `Postgres` for metadata, labels, inference records, and dataset tracking

The current system is file-based:

- API saves routed images under `data/lake`
- Label Studio reads local files from `data/lake`
- `sync_data.py` copies labeled images into `data/final/train`
- `retrain.py` trains from `data/final`

The new design should keep the same user-facing flow, but move the source of truth to the database.

## What Is Wrong With `WeirdHazelnut_database`
The collaborator folder is not a usable database integration for this project.

Observed issues:

- It is mainly a standalone classifier project copy, not an integrated data layer.
- It contains large amounts of dataset and artifact files, not a reusable MinIO/Postgres design.
- It does not define a clean metadata schema for inference, annotation, or dataset versioning.
- It does not fit the current project structure in `src/weird_hazelnut/`.

Conclusion: do not merge that folder as-is. Use it only as loose reference for training ideas if needed, but not as the implementation base.

## Current Repo Context
Relevant files in this repo:

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [README.md](README.md)
- [src/weird_hazelnut/pipeline/core.py](src/weird_hazelnut/pipeline/core.py)
- [src/weird_hazelnut/data/sync_worker.py](src/weird_hazelnut/data/sync_worker.py)
- [src/weird_hazelnut/integrations/label_studio.py](src/weird_hazelnut/integrations/label_studio.py)
- [src/weird_hazelnut/api/app.py](src/weird_hazelnut/api/app.py)
- [src/weird_hazelnut/training/retrain.py](src/weird_hazelnut/training/retrain.py)
- `docker-compose.yml`
- `config.yaml`
- `config.docker.yaml`

Current behavior:

- `HazelnutPipeline.run()` classifies an image and saves it locally.
- `LabelStudioClient.create_task()` points Label Studio at local file paths.
- `DataSyncWorker.sync()` copies completed labels into `data/final/train/<class>`.
- Training expects folder-based data.

## Target Architecture
Use Postgres as metadata source of truth and MinIO as the binary store.

### High-level flow

1. API receives an image upload.
2. The image is hashed and uploaded to MinIO.
3. Postgres stores image metadata and the inference result.
4. Routing still happens as `normal`, `uncertain`, or `anomalies`.
5. Uncertain images create Label Studio tasks.
6. When annotations are completed, sync job writes the label metadata into Postgres.
7. Training exports a versioned folder dataset from Postgres + MinIO into `data/final`.
8. Existing retraining scripts continue to read the exported folder structure.

### Ownership rules

- MinIO stores immutable image objects.
- Postgres stores metadata, labels, tasks, inference records, and dataset membership.
- Local folders become derived cache/export locations, not the canonical storage.

## Proposed Postgres Schema
Use UUID primary keys for app-owned rows.

### `images`
One row per uploaded image.

Fields:

- `id`
- `sha256`
- `original_filename`
- `content_type`
- `width`
- `height`
- `size_bytes`
- `minio_bucket`
- `minio_object_key`
- `storage_stage`
- `created_at`

Purpose:

- deduplicate uploads
- track where the object lives in MinIO
- record the current lifecycle state

### `inference_runs`
One row per prediction request.

Fields:

- `id`
- `image_id`
- `anomaly_score`
- `anomaly_predicted`
- `uncertain`
- `routed_category`
- `classifier_called`
- `classifier_label`
- `classifier_confidence`
- `ad_latency_ms`
- `classifier_latency_ms`
- `total_latency_ms`
- `mlflow_run_id`
- `created_at`

Purpose:

- trace each prediction
- connect model output to stored images
- support analytics and auditability

### `label_tasks`
One row for each Label Studio task created from an uncertain image.

Fields:

- `id`
- `image_id`
- `label_studio_task_id`
- `label_studio_project_id`
- `status`
- `created_at`
- `updated_at`

Purpose:

- track Label Studio task lifecycle
- avoid duplicated task creation

### `annotations`
Final human labels from Label Studio or imports.

Fields:

- `id`
- `image_id`
- `label_task_id`
- `source`
- `quality_label`
- `is_anomaly`
- `annotator_id`
- `label_studio_annotation_id`
- `raw_annotation`
- `created_at`

Purpose:

- store the canonical training label
- keep the raw annotation payload for traceability

### `dataset_versions`
Logical dataset snapshots used for training.

Fields:

- `id`
- `name`
- `description`
- `split_strategy`
- `created_by`
- `created_at`

Purpose:

- make training reproducible
- preserve which labeled samples were used in a training run

### `dataset_items`
Membership of images in a dataset version.

Fields:

- `dataset_version_id`
- `image_id`
- `annotation_id`
- `split`
- `label`
- `exported_path`

Purpose:

- map source records into train/val/test splits
- remember the folder export path used for retraining

### Suggested indexes

- `images.sha256`
- `images.storage_stage`
- `inference_runs.image_id`
- `label_tasks.image_id`
- `annotations.image_id`
- `annotations.quality_label`
- `dataset_items.dataset_version_id, split, label`

## MinIO Object Layout
Use deterministic object keys so the same image always maps the same way.

Recommended layout:

- `raw/YYYY/MM/DD/<sha256>.png`
- optionally `routed/normal/...`, `routed/uncertain/...`, `routed/anomalies/...` as metadata tags only, not duplicate objects

Recommendation:

- store the binary once
- represent routing and label state in Postgres
- avoid duplicating the same file under multiple object keys unless there is a strong reason

## Integration Changes By Component

### API / pipeline

- `POST /predict` should:
  - read image bytes
  - compute hash and basic image metadata
  - upload to MinIO
  - create or reuse the `images` row
  - run inference
  - insert `inference_runs`
  - create Label Studio task for uncertain images
- Keep the external API response shape stable where possible.
- Update `src/weird_hazelnut/pipeline/core.py` to stop treating local disk as the primary source of truth.

### Label Studio

- Continue creating Label Studio tasks for uncertain images.
- Prefer a path strategy that is compatible with Label Studio serving, but do not make local disk the canonical dataset store.
- Sync job should read completed annotations from Label Studio and persist them into Postgres.

### Sync worker

- Replace “copy files into `data/final/train`” as the main action.
- Sync worker should:
  - fetch completed tasks
  - map Label Studio labels into canonical project labels
  - write rows into `annotations`
  - mark task sync status in `label_tasks`
- File export should become a separate dataset export step.

### Training

- `retrain.py` should no longer assume `data/final` is the source of truth.
- It should:
  - create a new `dataset_version`
  - select labeled rows from Postgres
  - download corresponding images from MinIO
  - export them into the expected folder structure under `data/final`
  - call the existing sibling training projects using that export
- This keeps the current training code mostly unchanged while making the data layer authoritative.

## Implementation Phases

### Phase 1: Infrastructure

- Add `postgres` and `minio` to `docker-compose.yml`.
- Add environment/config entries for DB and object storage.
- Add required Python dependencies.
- Add init/bootstrap logic for:
  - database connection
  - MinIO bucket creation
  - initial schema migration

### Phase 2: Core storage layer

- Implement a database module for session management and repositories.
- Implement a storage module for MinIO upload/download helpers.
- Add Alembic migrations for the tables above.

### Phase 3: Pipeline integration

- Update inference flow to persist images and metadata.
- Update Label Studio task creation to reference stored images correctly.
- Store inference records in Postgres.

### Phase 4: Sync and dataset export

- Rework `sync_data.py` / `DataSyncWorker` to write annotations to Postgres.
- Add a dataset export helper that materializes a training folder tree from DB + MinIO.
- Update `retrain.py` to call the exporter before training.

### Phase 5: Validation

- Add tests for:
  - image deduplication by hash
  - object upload/download
  - annotation mapping
  - dataset export layout
  - basic pipeline persistence flow

## Suggested Table Review For You
Please review these design decisions before implementation:

1. Use `images` as the canonical object table with SHA256 deduplication.
2. Keep `inference_runs` separate from `images` so multiple predictions can exist for one image.
3. Store Label Studio task lifecycle in `label_tasks`, not in `images`.
4. Store the final training label in `annotations`, not in `label_tasks`.
5. Use `dataset_versions` and `dataset_items` for reproducible retraining.

If you want a simpler schema, the minimum viable version can be reduced to:

- `images`
- `annotations`
- `dataset_versions`
- `dataset_items`

But the fuller schema is better for auditability and future expansion.

## Acceptance Criteria

- API can ingest an image and persist it to MinIO + Postgres.
- Uncertain predictions create Label Studio tasks and are tracked in Postgres.
- Completed labels sync into Postgres successfully.
- Retraining exports a dataset from Postgres + MinIO and trains using the exported folder tree.
- Existing project behavior remains stable for inference responses and training inputs.

## Assumptions

- MLflow stays separate from the new data layer.
- Existing sibling training repositories remain unchanged.
- Local `data/final` remains as an export target for compatibility.
- Canonical class labels are `good`, `crack`, `cut`, `hole`, and `print`.
- `WeirdHazelnut_database` is not migrated directly; it is only used as background reference.
