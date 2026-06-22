"""
Supabase Database Operations for ContractIQ.
Includes mock fallbacks if Supabase is not active.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from logger import GLOBAL_LOGGER as log
from src.database.supabase_client import is_supabase_active, supabase_client

# Local fallback directory for mock mode
LOCAL_DB_DIR = Path("data/local_db")
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)

# File paths for mock data
MOCK_PROFILES_FILE = LOCAL_DB_DIR / "profiles.json"
MOCK_SESSIONS_FILE = LOCAL_DB_DIR / "sessions.json"
MOCK_EXTRACTIONS_FILE = LOCAL_DB_DIR / "extractions.json"
MOCK_USAGE_FILE = LOCAL_DB_DIR / "usage.json"


def _load_mock_file(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_mock_file(path: Path, data: dict):
    try:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        log.error("Failed to write mock database file", path=str(path), error=str(e))


# ============================================================
# Profiles
# ============================================================

def create_profile_if_not_exists(user_id: str, email: str) -> bool:
    """Create a profile for the user in the profiles table."""
    if is_supabase_active():
        try:
            # Check if profile already exists
            res = supabase_client.table("profiles").select("id").eq("id", user_id).execute()
            if not res.data:
                supabase_client.table("profiles").insert({
                    "id": user_id,
                    "email": email,
                    "subscription_tier": "free"
                }).execute()
                log.info("Supabase user profile created", user_id=user_id, email=email)
            return True
        except Exception as e:
            log.error("Failed to create profile in Supabase", user_id=user_id, error=str(e))
            return False
    else:
        # Mock implementation
        db = _load_mock_file(MOCK_PROFILES_FILE)
        if user_id not in db:
            db[user_id] = {
                "id": user_id,
                "email": email,
                "subscription_tier": "free",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            _save_mock_file(MOCK_PROFILES_FILE, db)
            log.info("Mock user profile created", user_id=user_id, email=email)
        return True


# ============================================================
# Usage Logs & Limits
# ============================================================

def log_usage(user_id: str, action: str, pages_processed: int = 0, tokens_used: int = 0) -> bool:
    """Log an API transaction and page/token count."""
    if is_supabase_active():
        try:
            supabase_client.table("usage_logs").insert({
                "user_id": user_id,
                "action": action,
                "pages_processed": pages_processed,
                "tokens_used": tokens_used
            }).execute()
            log.info("Logged usage in Supabase", user_id=user_id, action=action, pages=pages_processed)
            return True
        except Exception as e:
            log.error("Failed to log usage in Supabase", user_id=user_id, error=str(e))
            return False
    else:
        # Mock implementation
        db = _load_mock_file(MOCK_USAGE_FILE)
        logs = db.setdefault(user_id, [])
        logs.append({
            "action": action,
            "pages_processed": pages_processed,
            "tokens_used": tokens_used,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        _save_mock_file(MOCK_USAGE_FILE, db)
        log.info("Logged usage in Mock DB", user_id=user_id, action=action, pages=pages_processed)
        return True


def check_user_limits(user_id: str, monthly_limit: int = 50) -> bool:
    """
    Check if the user has exceeded their monthly processing page limits.
    Returns True if allowed (under limit), False if blocked.
    """
    first_day_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    if is_supabase_active():
        try:
            # Check user subscription tier (premium users get unlimited/higher limits)
            prof = supabase_client.table("profiles").select("subscription_tier").eq("id", user_id).execute()
            if prof.data and prof.data[0].get("subscription_tier") == "premium":
                return True
                
            # Aggregate pages processed in the current month
            res = supabase_client.table("usage_logs")\
                .select("pages_processed")\
                .eq("user_id", user_id)\
                .gte("created_at", first_day_of_month.isoformat())\
                .execute()
                
            total_pages = sum(item.get("pages_processed", 0) for item in res.data)
            log.info("Monthly usage check", user_id=user_id, total_pages=total_pages, limit=monthly_limit)
            return total_pages < monthly_limit
        except Exception as e:
            log.error("Failed to query usage limits in Supabase, defaulting to True", user_id=user_id, error=str(e))
            return True
    else:
        # Mock implementation
        db = _load_mock_file(MOCK_PROFILES_FILE)
        prof = db.get(user_id, {})
        if prof.get("subscription_tier") == "premium":
            return True

        usage_db = _load_mock_file(MOCK_USAGE_FILE)
        logs = usage_db.get(user_id, [])
        total_pages = 0
        for log_entry in logs:
            try:
                dt = datetime.fromisoformat(log_entry.get("created_at"))
                if dt >= first_day_of_month:
                    total_pages += log_entry.get("pages_processed", 0)
            except Exception:
                pass
        log.info("Mock monthly usage check", user_id=user_id, total_pages=total_pages, limit=monthly_limit)
        return total_pages < monthly_limit


# ============================================================
# RAG Sessions
# ============================================================

def save_rag_session(session_id: str, user_id: str, metadata: dict) -> bool:
    """Save RAG session configuration/metadata."""
    if is_supabase_active():
        try:
            supabase_client.table("rag_sessions").upsert({
                "id": session_id,
                "user_id": user_id,
                "metadata": metadata
            }).execute()
            log.info("Saved RAG session to Supabase", session_id=session_id)
            return True
        except Exception as e:
            log.error("Failed to save RAG session in Supabase", session_id=session_id, error=str(e))
            return False
    else:
        # Mock implementation
        db = _load_mock_file(MOCK_SESSIONS_FILE)
        db[session_id] = {
            "user_id": user_id,
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        _save_mock_file(MOCK_SESSIONS_FILE, db)
        
        # Also backup to Redis if available
        try:
            from src.cache.redis_cache import cache
            if cache.is_available:
                cache.store_rag_session(session_id, metadata)
        except Exception:
            pass
            
        log.info("Saved RAG session to Mock DB", session_id=session_id)
        return True


def get_rag_session(session_id: str) -> Optional[dict]:
    """Retrieve RAG session metadata."""
    if is_supabase_active():
        try:
            res = supabase_client.table("rag_sessions").select("metadata").eq("id", session_id).execute()
            if res.data:
                return res.data[0].get("metadata")
            return None
        except Exception as e:
            log.error("Failed to retrieve RAG session from Supabase", session_id=session_id, error=str(e))
            return None
    else:
        # Check Redis first as cache fallback
        try:
            from src.cache.redis_cache import cache
            if cache.is_available:
                sess = cache.get_rag_session(session_id)
                if sess:
                    return sess
        except Exception:
            pass

        # Mock implementation
        db = _load_mock_file(MOCK_SESSIONS_FILE)
        session_info = db.get(session_id)
        if session_info:
            return session_info.get("metadata")
        return None


# ============================================================
# Key-Term Extraction Logs
# ============================================================

def save_extraction_results(
    session_id: str, 
    user_id: str, 
    extracted_terms: dict, 
    confidence_scores: dict
) -> bool:
    """Save pipeline structured key-term extractions to the database."""
    if is_supabase_active():
        try:
            supabase_client.table("extractions").insert({
                "session_id": session_id,
                "user_id": user_id,
                "extracted_terms": extracted_terms,
                "confidence_scores": confidence_scores
            }).execute()
            log.info("Saved extraction results to Supabase", session_id=session_id)
            return True
        except Exception as e:
            log.error("Failed to save extraction results to Supabase", session_id=session_id, error=str(e))
            return False
    else:
        # Mock implementation
        db = _load_mock_file(MOCK_EXTRACTIONS_FILE)
        extractions = db.setdefault(session_id, [])
        extractions.append({
            "user_id": user_id,
            "extracted_terms": extracted_terms,
            "confidence_scores": confidence_scores,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        _save_mock_file(MOCK_EXTRACTIONS_FILE, db)
        log.info("Saved extraction results to Mock DB", session_id=session_id)
        return True
