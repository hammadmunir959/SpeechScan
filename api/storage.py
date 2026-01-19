import os
import logging
import boto3
from botocore.exceptions import ClientError
from api.config import settings

logger = logging.getLogger(__name__)

class StorageProvider:
    def upload(self, local_path: str, remote_name: str) -> str:
        raise NotImplementedError()

    def delete(self, remote_name: str):
        raise NotImplementedError()

class LocalStorage(StorageProvider):
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def upload(self, local_path: str, remote_name: str) -> str:
        dest_path = os.path.join(self.upload_dir, remote_name)
        if local_path != dest_path:
            import shutil
            shutil.copy2(local_path, dest_path)
        return dest_path

    def delete(self, remote_name: str):
        path = os.path.join(self.upload_dir, remote_name)
        if os.path.exists(path):
            os.remove(path)

class S3Storage(StorageProvider):
    def __init__(self, bucket: str, endpoint_url: str = None):
        self.bucket = bucket
        self.s3 = boto3.client('s3', endpoint_url=endpoint_url)

    def upload(self, local_path: str, remote_name: str) -> str:
        try:
            self.s3.upload_file(local_path, self.bucket, remote_name)
            return f"s3://{self.bucket}/{remote_name}"
        except ClientError as e:
            logger.error(f"S3 Upload error: {e}")
            return None

    def delete(self, remote_name: str):
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=remote_name)
        except ClientError as e:
            logger.error(f"S3 Delete error: {e}")

def get_storage() -> StorageProvider:
    """
    Factory to get the configured storage provider.
    """
    # For now, default to local storage but allow S3 via env
    if os.getenv("S3_BUCKET"):
        return S3Storage(
            bucket=os.getenv("S3_BUCKET"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL") # Useful for MinIO
        )
    return LocalStorage()
