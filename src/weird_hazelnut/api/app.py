import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from weird_hazelnut.config import load_config
from weird_hazelnut.integrations import LabelStudioClient
from weird_hazelnut.integrations.mlflow_tracking import MlflowTracker
from weird_hazelnut.integrations.mlflow_tracking import resolve_model_paths
from weird_hazelnut.pipeline import HazelnutPipeline


class AppState:
    pipeline: HazelnutPipeline | None = None
    tracker: MlflowTracker | None = None


state = AppState()


def create_app() -> FastAPI:
    config = load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            ad_model_path, cls_model_path, cls_meta_path = resolve_model_paths(config)
            ls_client = _create_label_studio_client(config)
            state.pipeline = HazelnutPipeline(
                anomaly_model_path=ad_model_path,
                classifier_model_path=cls_model_path,
                classifier_meta_path=cls_meta_path,
                threshold_low=config["pipeline"]["thresholds"]["low"],
                threshold_high=config["pipeline"]["thresholds"]["high"],
                lake_dir=config["pipeline"]["lake_dir"],
                ls_client=ls_client,
                anomaly_device=config["models"]["anomaly_detector"].get("device", "CPU"),
            )
            print("Pipeline initialized successfully.")

            state.tracker = MlflowTracker(config)
            state.tracker.start()
            yield
        finally:
            if state.tracker:
                state.tracker.end()
                print("MLflow run ended.")

    app = FastAPI(title="WeirdHazelnut Combined API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "pipeline_loaded": state.pipeline is not None,
            "mlflow_active": state.tracker is not None and state.tracker.run is not None,
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if not state.pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            start_time = time.perf_counter()
            result = state.pipeline.run(image)
            total_latency = (time.perf_counter() - start_time) * 1000
            result["total_latency_ms"] = total_latency

            if state.tracker:
                state.tracker.log_prediction(image, result, total_latency)

            response = dict(result)
            response.pop("heat_map", None)
            response.pop("anomaly_map", None)
            return response
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def _create_label_studio_client(config: dict) -> LabelStudioClient | None:
    ls_cfg = config.get("label_studio", {})
    if not ls_cfg:
        return None
    return LabelStudioClient(
        url=ls_cfg.get("url"),
        api_key=ls_cfg.get("api_key"),
        project_id=ls_cfg.get("project_id"),
    )


app = create_app()
