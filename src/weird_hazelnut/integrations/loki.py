import logging
import queue
import sys
import threading
import time
from typing import Any, Dict

import requests


class LokiHandler(logging.Handler):
    """A custom logging handler that pushes log records to Grafana Loki in a background thread."""

    def __init__(
        self,
        url: str,
        user: str = "",
        password: str = "",
        labels: Dict[str, str] = None,
        batch_interval: float = 2.0,
        max_batch_size: int = 50,
    ):
        super().__init__()
        # Clean URL: ensure it ends with /loki/api/v1/push
        if not url.endswith("/loki/api/v1/push"):
            url = url.rstrip("/") + "/loki/api/v1/push"
        self.url = url
        self.user = user
        self.password = password
        self.labels = labels or {"app": "weird-hazelnut"}
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size

        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            ts_ns = str(int(record.created * 1e9))
            self.queue.put((ts_ns, msg))
        except Exception:
            self.handleError(record)

    def _worker(self) -> None:
        batch = []
        last_flush = time.time()

        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Wait for an item with a timeout
                ts_ns, msg = self.queue.get(timeout=0.1)
                batch.append((ts_ns, msg))
                self.queue.task_done()
            except queue.Empty:
                pass

            # Check if we should flush
            now = time.time()
            if batch and (
                len(batch) >= self.max_batch_size
                or (now - last_flush) >= self.batch_interval
            ):
                self._flush(batch)
                batch = []
                last_flush = now

        # Flush any remaining items before exiting
        if batch:
            self._flush(batch)

    def _flush(self, batch: list) -> None:
        payload = {"streams": [{"stream": self.labels, "values": batch}]}

        auth = None
        if self.user and self.password:
            auth = (self.user, self.password)

        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.url, json=payload, auth=auth, headers=headers, timeout=5
            )
            if response.status_code != 204:
                print(
                    f"LokiHandler failed to push logs (status {response.status_code}): {response.text}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"LokiHandler error sending logs to Loki: {e}", file=sys.stderr)

    def close(self) -> None:
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=3)
        super().close()


def setup_loki_logging(config: dict) -> None:
    """Setup cloud logging to Loki if configured."""
    cloud_cfg = config.get("cloud_logging", {})
    if not cloud_cfg.get("enabled", False):
        return

    url = cloud_cfg.get("loki_url")
    if not url:
        return

    user = cloud_cfg.get("loki_user", "")
    password = cloud_cfg.get("loki_password", "")
    labels = cloud_cfg.get("labels", {"app": "weird-hazelnut", "env": "production"})

    # Setup Loki handler
    loki_handler = LokiHandler(
        url=url,
        user=user,
        password=password,
        labels=labels,
    )
    
    # Formatter for Loki
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    )
    loki_handler.setFormatter(formatter)
    loki_handler.setLevel(logging.INFO)

    # Add handler to root logger
    root_logger = logging.getLogger()
    if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(loki_handler)

    # Add handler to uvicorn loggers to capture web requests and server events
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        u_logger = logging.getLogger(logger_name)
        u_logger.addHandler(loki_handler)

    print(f"Cloud logging initialized: pushing logs to Loki ({url}) with user={user}")
