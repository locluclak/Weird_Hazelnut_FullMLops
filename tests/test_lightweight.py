import tempfile
import unittest
from pathlib import Path

import src
from src.weird_hazelnut.config import load_config
from src.weird_hazelnut.data.database import Database
from src.weird_hazelnut.data.repositories import DataRepository
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

    def test_image_records_are_deduplicated_by_sha256(self):
        database = Database("sqlite+pysqlite:///:memory:")
        database.bootstrap()

        values = {
            "sha256": "a" * 64,
            "original_filename": "sample.png",
            "content_type": "image/png",
            "width": 32,
            "height": 32,
            "size_bytes": 128,
            "minio_bucket": "weird-hazelnut",
            "minio_object_key": "raw/2026/05/30/sample.png",
            "storage_stage": "raw",
        }

        with database.session() as session:
            repo = DataRepository(session)
            first = repo.get_or_create_image(values)
            second = repo.get_or_create_image({**values, "storage_stage": "uncertain"})

        self.assertEqual(first.id, second.id)
        self.assertEqual(second.storage_stage, "uncertain")

    def test_local_file_url_resolves_to_data_lake_path(self):
        worker = object.__new__(DataSyncWorker)

        resolved = worker._resolve_local_path(
            "/data/local-files/?d=lake/uncertain/sample.png"
        )

        self.assertTrue(resolved.endswith("data\\lake\\uncertain\\sample.png") or resolved.endswith("data/lake/uncertain/sample.png"))


if __name__ == "__main__":
    unittest.main()
