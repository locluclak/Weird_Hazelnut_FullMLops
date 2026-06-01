import hashlib
import io
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def object_key_for_hash(
    sha256: str,
    suffix: str = ".png",
    prefix: str | None = None,
) -> str:
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    if prefix:
        return f"{prefix.strip('/')}/{sha256}{ext.lower()}"
    now = datetime.utcnow()
    return f"raw/{now:%Y/%m/%d}/{sha256}{ext.lower()}"


class MinioObjectStore:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        region: str = "us-east-1",
        public_endpoint: str | None = None,
        public_secure: bool | None = None,
    ):
        from minio import Minio

        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self.public_client = (
            Minio(
                public_endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure if public_secure is None else public_secure,
                region=region,
            )
            if public_endpoint
            else self.client
        )

    def bootstrap(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload_bytes(
        self,
        object_key: str,
        content: bytes,
        content_type: str = "image/png",
    ) -> None:
        self.client.put_object(
            self.bucket,
            object_key,
            io.BytesIO(content),
            length=len(content),
            content_type=content_type,
        )

    def download_to_path(self, object_key: str, target_path: str | Path) -> None:
        response = self.client.get_object(self.bucket, object_key)
        try:
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as f:
                for chunk in response.stream(32 * 1024):
                    f.write(chunk)
        finally:
            response.close()
            response.release_conn()

    def get_bytes(self, object_key: str) -> bytes:
        response = self.client.get_object(self.bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def open_image(self, object_key: str) -> Image.Image:
        content = self.get_bytes(object_key)
        return Image.open(io.BytesIO(content)).convert("RGB")

    def presigned_get_url(self, object_key: str, expires_seconds: int = 7 * 24 * 3600) -> str:
        return self.public_client.presigned_get_object(
            self.bucket,
            object_key,
            expires=timedelta(seconds=expires_seconds),
        )
