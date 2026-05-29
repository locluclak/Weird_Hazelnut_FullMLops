import tempfile
import unittest
from pathlib import Path

import src
from src.weird_hazelnut.config import load_config
from src.weird_hazelnut.data.sync_worker import DataSyncWorker
from src.weird_hazelnut.integrations.label_studio import _to_label_studio_path


class LightweightBehaviorTests(unittest.TestCase):
    def test_load_config_from_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("pipeline:\n  thresholds:\n    low: 0.1\n", encoding="utf-8")

            config = load_config(config_path)

        self.assertEqual(config["pipeline"]["thresholds"]["low"], 0.1)

    def test_label_studio_path_maps_data_lake_to_local_file_uri(self):
        image_path = "D:/repo/WeirdHazelnut/data/lake/uncertain/sample.png"

        self.assertEqual(
            _to_label_studio_path(image_path),
            "lake/uncertain/sample.png",
        )

    def test_label_studio_local_file_url_matches_document_root(self):
        image_path = "D:/repo/WeirdHazelnut/data/lake/uncertain/sample.png"
        relative_path = _to_label_studio_path(image_path)

        self.assertEqual(
            f"/data/local-files/?d={relative_path}",
            "/data/local-files/?d=lake/uncertain/sample.png",
        )

    def test_extract_label_returns_anomaly_subclass_when_present(self):
        worker = object.__new__(DataSyncWorker)
        result = [
            {"from_name": "sentiment", "value": {"choices": ["Anomaly"]}},
            {"from_name": "label", "value": {"choices": ["crack"]}},
        ]

        self.assertEqual(worker._extract_label(result), "crack")

    def test_extract_label_returns_normal_primary_choice(self):
        worker = object.__new__(DataSyncWorker)
        result = [{"from_name": "sentiment", "value": {"choices": ["Normal"]}}]

        self.assertEqual(worker._extract_label(result), "Normal")


if __name__ == "__main__":
    unittest.main()
