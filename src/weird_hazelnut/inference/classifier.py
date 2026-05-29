import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class HazelnutClassifier:
    def __init__(self, onnx_path: str, meta_path: str):
        self.onnx_path = Path(onnx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.classes = self.meta["classes"]
        self.img_size = self.meta["img_size"]

        opts = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: Image.Image):
        image = image.convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_data = np.array(image).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std

        img_data = img_data.transpose(2, 0, 1)
        return np.expand_dims(img_data, axis=0).astype(np.float32)

    def predict(self, image: Image.Image) -> dict:
        start_time = time.perf_counter()
        img_data = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: img_data})
        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        idx = int(np.argmax(probs))

        latency = (time.perf_counter() - start_time) * 1000

        return {
            "label": self.classes[idx],
            "confidence": float(probs[idx]),
            "latency_ms": latency,
        }

