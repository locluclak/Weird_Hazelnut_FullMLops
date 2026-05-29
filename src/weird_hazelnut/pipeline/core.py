import os
import uuid
from datetime import datetime

from PIL import Image

from weird_hazelnut.inference import AnomalyDetector, HazelnutClassifier
from weird_hazelnut.integrations.label_studio import LabelStudioClient


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
        anomaly_device: str = "CPU",
    ):
        self.anomaly_detector = AnomalyDetector(anomaly_model_path, device=anomaly_device)
        self.classifier = HazelnutClassifier(classifier_model_path, classifier_meta_path)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.lake_dir = lake_dir
        self.ls_client = ls_client

        for d in ["normal", "uncertain", "anomalies"]:
            os.makedirs(os.path.join(self.lake_dir, d), exist_ok=True)

    def run(self, image: Image.Image) -> dict:
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
        save_path = self.save_image(image, category, score, result["label"])

        if category == "uncertain" and self.ls_client:
            result["ls_task_id"] = self.ls_client.create_task(os.path.abspath(save_path), score)

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
