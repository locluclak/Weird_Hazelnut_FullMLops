import os
import time
import uuid
from datetime import datetime

from PIL import Image

from weird_hazelnut.data.layer import DataLayer
from weird_hazelnut.inference import AnomalyDetector, HazelnutClassifier
from weird_hazelnut.integrations.label_studio import LabelStudioClient, local_file_url


class HazelnutPipeline:
    def __init__(
        self,
        anomaly_model_path: str,
        classifier_model_path: str,
        classifier_meta_path: str,
        threshold_low: float = 0.3,
        threshold_high: float = 0.7,
        lake_dir: str = "data/lake",
        ls_client: LabelStudioClient | None = None,
        data_layer: DataLayer | None = None,
        anomaly_device: str = "CPU",
    ):
        self.anomaly_detector = AnomalyDetector(anomaly_model_path, device=anomaly_device)
        self.classifier = HazelnutClassifier(classifier_model_path, classifier_meta_path)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.lake_dir = lake_dir
        self.ls_client = ls_client
        self.data_layer = data_layer

        if not self.data_layer:
            for d in ["normal", "uncertain", "anomalies"]:
                os.makedirs(os.path.join(self.lake_dir, d), exist_ok=True)

    def run(
        self,
        image: Image.Image,
        image_bytes: bytes | None = None,
        filename: str | None = None,
        content_type: str | None = None,
        mlflow_run_id: str | None = None,
        persist: bool = True,
    ) -> dict:
        start_time = time.perf_counter()
        image_record = None
        if self.data_layer and persist:
            image_record = self.data_layer.persist_image(
                image=image,
                image_bytes=image_bytes,
                original_filename=filename,
                content_type=content_type,
            )

        ad_result = self.anomaly_detector.predict(image)
        score = ad_result["anomaly_score"]
        is_anomaly = ad_result["is_anomaly"]
        uncertain = self.threshold_low <= score <= self.threshold_high

        result = {
            "anomaly": is_anomaly,
            "anomaly_score": score,
            "uncertain": uncertain,
            "ad_latency_ms": ad_result["latency_ms"],
            "model_b_called": False,
            "classifier_latency_ms": 0.0,
            "label": None,
            "anomaly_map": ad_result.get("anomaly_map"),
            "heat_map": ad_result.get("heat_map"),
            "ls_task_id": None,
        }

        if is_anomaly:
            result["model_b_called"] = True
            cls_result = self.classifier.predict(image)
            result["label"] = cls_result["label"]
            result["confidence"] = cls_result["confidence"]
            result["classifier_latency_ms"] = cls_result["latency_ms"]

        category = self.route_category(score)
        if self.data_layer and image_record:
            self.data_layer.update_image_stage(image_record.id, category)

        save_path = None
        if not self.data_layer:
            save_path = self.save_image(image, category, score, result["label"])

        if category == "uncertain" and self.ls_client and (image_record or not self.data_layer):
            image_url = None
            if self.data_layer and image_record:
                image_url = self.data_layer.presigned_image_url(
                    image_record.minio_object_key,
                )
            else:
                image_url = local_file_url(os.path.abspath(save_path))

            result["ls_task_id"] = self.ls_client.create_task(
                image_url,
                score,
                metadata={
                    "image_id": image_record.id if image_record else None,
                    "minio_bucket": image_record.minio_bucket if image_record else None,
                    "minio_object_key": image_record.minio_object_key if image_record else None,
                },
            )
            if self.data_layer and image_record and result["ls_task_id"]:
                self.data_layer.record_label_task(
                    image_id=image_record.id,
                    label_studio_task_id=result["ls_task_id"],
                    label_studio_project_id=self.ls_client.project_id,
                )

        result["image_id"] = image_record.id if image_record else None

        if self.data_layer and image_record:
            self.data_layer.record_inference(
                {
                    "image_id": image_record.id,
                    "anomaly_score": score,
                    "anomaly_predicted": is_anomaly,
                    "uncertain": uncertain,
                    "routed_category": category,
                    "classifier_called": result["model_b_called"],
                    "classifier_label": result["label"],
                    "classifier_confidence": result.get("confidence"),
                    "ad_latency_ms": result["ad_latency_ms"],
                    "classifier_latency_ms": result["classifier_latency_ms"],
                    "total_latency_ms": (time.perf_counter() - start_time) * 1000,
                    "mlflow_run_id": mlflow_run_id,
                }
            )

        return result

    def route_category(self, score: float) -> str:
        if score > self.threshold_high:
            return "anomalies"
        if score >= self.threshold_low:
            return "uncertain"
        return "normal"

    def save_image(self, image: Image.Image, category: str, score: float, label: str | None = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_score{score:.2f}.png"
        if label:
            filename = f"{timestamp}_{unique_id}_score{score:.2f}_{label}.png"

        save_path = os.path.join(self.lake_dir, category, filename)
        image.save(save_path)
        return save_path
