import pathlib
import time

import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer
from PIL import Image


class AnomalyDetector:
    def __init__(self, model_xml_path: str, device: str = "CPU"):
        self.path = pathlib.Path(model_xml_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"OpenVINO Model XML file not found at: {self.path}")

        self.inferencer = OpenVINOInferencer(path=str(self.path), device=device)

    def predict(self, image: Image.Image) -> dict:
        img_np = np.array(image.convert("RGB"))
        img_resized = cv2.resize(img_np, (256, 256))

        start_time = time.perf_counter()
        raw_pred = self.inferencer.predict(image=img_resized)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if isinstance(raw_pred, (list, tuple)):
            raw_pred = raw_pred[0]

        pred_score = _get_field(raw_pred, "pred_score")
        pred_label = _get_field(raw_pred, "pred_label")
        anomaly_map = _get_field(raw_pred, "anomaly_map")
        heat_map = _get_field(raw_pred, "heat_map")

        score = 0.0
        if pred_score is not None:
            score = float(_to_numpy(pred_score).item())

        is_anomaly = False
        if pred_label is not None:
            is_anomaly = bool(_to_numpy(pred_label).item())

        anomaly_map_np = _to_numpy(anomaly_map)
        if anomaly_map_np is not None:
            anomaly_map_np = anomaly_map_np.squeeze()

        if heat_map is None and anomaly_map_np is not None:
            am_min, am_max = anomaly_map_np.min(), anomaly_map_np.max()
            am_norm = (anomaly_map_np - am_min) / (am_max - am_min + 1e-8)
            am_uint8 = (am_norm * 255).astype(np.uint8)
            heat_map_bgr = cv2.applyColorMap(am_uint8, cv2.COLORMAP_JET)
            heat_map = cv2.cvtColor(heat_map_bgr, cv2.COLOR_BGR2RGB)
        else:
            heat_map = _to_numpy(heat_map)
            if heat_map is not None:
                heat_map = heat_map.squeeze()

        return {
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "anomaly_map": anomaly_map_np,
            "heat_map": heat_map,
            "latency_ms": latency_ms,
        }


def _get_field(obj, attr_name):
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    if isinstance(obj, dict):
        return obj.get(attr_name)
    return None


def _to_numpy(arr):
    if arr is None:
        return None
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.array(arr)

