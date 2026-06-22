"""
Supabase Storage Operations for ContractIQ.
Includes mock fallbacks if Supabase is not active.
"""
import os
from pathlib import Path
from logger import GLOBAL_LOGGER as log
from src.database.supabase_client import is_supabase_active, supabase_client

# Local fallback directory for mock storage
LOCAL_STORAGE_DIR = Path("data/local_storage")
LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def upload_file_to_supabase(bucket: str, path: str, file_bytes: bytes) -> str:
    """
    Upload a file to Supabase Storage.
    Returns the path in the bucket.
    """
    # Normalize path separator for cloud storage (always use forward slashes)
    path = path.replace("\\", "/")
    
    if is_supabase_active():
        try:
            try:
                # Try upserting the file
                supabase_client.storage.from_(bucket).upload(
                    path=path,
                    file=file_bytes,
                    file_options={"x-upsert": "true"}
                )
            except Exception:
                try:
                    # Fallback to direct upload
                    supabase_client.storage.from_(bucket).upload(path, file_bytes)
                except Exception:
                    # Fallback to update if it already exists
                    supabase_client.storage.from_(bucket).update(path, file_bytes)
            
            log.info("File uploaded to Supabase Storage", bucket=bucket, path=path)
            return path
        except Exception as e:
            log.error("Failed to upload file to Supabase Storage", bucket=bucket, path=path, error=str(e))
            raise e
    else:
        # Mock fallback
        local_path = LOCAL_STORAGE_DIR / bucket / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(file_bytes)
        log.info("File saved to Mock Storage", path=str(local_path))
        return str(local_path)


def download_file_from_supabase(bucket: str, path: str) -> bytes:
    """
    Download a file from Supabase Storage.
    Returns the file bytes.
    """
    path = path.replace("\\", "/")
    
    if is_supabase_active():
        try:
            res = supabase_client.storage.from_(bucket).download(path)
            log.info("File downloaded from Supabase Storage", bucket=bucket, path=path)
            return res
        except Exception as e:
            log.error("Failed to download file from Supabase Storage", bucket=bucket, path=path, error=str(e))
            raise e
    else:
        # Mock fallback
        local_path = LOCAL_STORAGE_DIR / bucket / path
        if local_path.exists():
            log.info("File read from Mock Storage", path=str(local_path))
            return local_path.read_bytes()
        
        # Check standard document analysis folder too as fallback
        fallback_local = Path("data/document_analysis") / path
        if fallback_local.exists():
            log.info("File read from Mock Document Analysis directory", path=str(fallback_local))
            return fallback_local.read_bytes()
            
        raise FileNotFoundError(f"File not found in storage: {path}")


def delete_file_from_supabase(bucket: str, path: str) -> bool:
    """
    Delete a file from Supabase Storage.
    """
    path = path.replace("\\", "/")
    
    if is_supabase_active():
        try:
            supabase_client.storage.from_(bucket).remove([path])
            log.info("File deleted from Supabase Storage", bucket=bucket, path=path)
            return True
        except Exception as e:
            log.error("Failed to delete file from Supabase Storage", bucket=bucket, path=path, error=str(e))
            return False
    else:
        # Mock fallback
        local_path = LOCAL_STORAGE_DIR / bucket / path
        if local_path.exists():
            local_path.unlink()
            log.info("File deleted from Mock Storage", path=str(local_path))
            return True
        return False
