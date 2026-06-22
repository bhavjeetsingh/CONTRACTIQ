"""
Unit tests for Supabase Integration & Usage Tracking in Mock Mode.
"""
import os
import shutil
import pytest
from pathlib import Path
from src.database.supabase_client import is_supabase_active
from src.database.supabase_db import (
    create_profile_if_not_exists,
    log_usage,
    check_user_limits,
    save_rag_session,
    get_rag_session,
    save_extraction_results,
    LOCAL_DB_DIR
)
from src.database.supabase_storage import (
    upload_file_to_supabase,
    download_file_from_supabase,
    delete_file_from_supabase,
    LOCAL_STORAGE_DIR
)
from src.document_ingestion.data_ingestion import DocHandler


def test_supabase_inactive_by_default():
    # Verify that without env vars, is_supabase_active is False
    assert is_supabase_active() is False


def test_mock_profile_creation():
    user_id = "test-user-123"
    email = "test@example.com"
    
    # Ensure any old file is removed
    profiles_file = LOCAL_DB_DIR / "profiles.json"
    if profiles_file.exists():
        profiles_file.unlink()
        
    assert create_profile_if_not_exists(user_id, email) is True
    assert profiles_file.exists()


def test_mock_usage_logging_and_limits():
    user_id = "test-user-123"
    
    # Reset usage mock file
    usage_file = LOCAL_DB_DIR / "usage.json"
    if usage_file.exists():
        usage_file.unlink()
        
    # Initially should be well under the limit
    assert check_user_limits(user_id, monthly_limit=5) is True
    
    # Log usage of 3 pages
    assert log_usage(user_id, action="analyze", pages_processed=3) is True
    assert check_user_limits(user_id, monthly_limit=5) is True
    
    # Log usage of another 3 pages (total 6)
    assert log_usage(user_id, action="extract", pages_processed=3) is True
    # Now should exceed limit of 5
    assert check_user_limits(user_id, monthly_limit=5) is False
    
    # Unlimited for premium
    profiles_file = LOCAL_DB_DIR / "profiles.json"
    import json
    db = json.loads(profiles_file.read_text(encoding="utf-8"))
    db[user_id]["subscription_tier"] = "premium"
    profiles_file.write_text(json.dumps(db), encoding="utf-8")
    
    assert check_user_limits(user_id, monthly_limit=5) is True


def test_mock_storage_operations():
    bucket = "test-bucket"
    filename = "subfolder/hello.txt"
    file_bytes = b"Hello from ContractIQ Mock Storage!"
    
    # Ensure old bucket directory is clean
    bucket_dir = LOCAL_STORAGE_DIR / bucket
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
        
    # Upload
    path = upload_file_to_supabase(bucket, filename, file_bytes)
    assert path is not None
    
    # Download
    downloaded = download_file_from_supabase(bucket, filename)
    assert downloaded == file_bytes
    
    # Delete
    assert delete_file_from_supabase(bucket, filename) is True
    assert not (LOCAL_STORAGE_DIR / bucket / filename).exists()


def test_page_count_estimation():
    dh = DocHandler(session_id="test-pages-session")
    
    # Single page formats (txt)
    temp_txt = Path(dh.session_path) / "test.txt"
    temp_txt.write_text("Hello world", encoding="utf-8")
    assert dh.get_page_count(str(temp_txt)) == 1
    
    # Cleanup temp
    if temp_txt.exists():
        temp_txt.unlink()
    shutil.rmtree(dh.session_path, ignore_errors=True)
