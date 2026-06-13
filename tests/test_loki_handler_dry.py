import unittest
import logging
from src.weird_hazelnut.integrations.loki import LokiHandler, setup_loki_logging

class TestLokiHandlerDry(unittest.TestCase):
    def test_loki_handler_initialization(self):
        handler = LokiHandler(
            url="https://logs-prod-020.grafana.net/loki/api/v1/push",
            user="1647033",
            password="test-password",
            labels={"app": "test-app", "env": "testing"}
        )
        self.assertEqual(handler.url, "https://logs-prod-020.grafana.net/loki/api/v1/push")
        self.assertEqual(handler.user, "1647033")
        self.assertEqual(handler.password, "test-password")
        self.assertEqual(handler.labels["app"], "test-app")
        
        # Test clean shutdown
        handler.close()

    def test_setup_loki_logging_disabled(self):
        config = {
            "cloud_logging": {
                "enabled": False,
                "loki_url": "https://logs-prod-020.grafana.net/loki/api/v1/push"
            }
        }
        # Should not raise any errors and should not add handler
        setup_loki_logging(config)
        root_logger = logging.getLogger()
        has_loki = any(isinstance(h, LokiHandler) for h in root_logger.handlers)
        self.assertFalse(has_loki)

if __name__ == "__main__":
    unittest.main()
