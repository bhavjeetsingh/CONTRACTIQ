"""
Supabase client initializer for ContractIQ.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from logger import GLOBAL_LOGGER as log

# Ensure env is loaded
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
# Optional JWT secret for local token decoding (adds speed)
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# Initialize client if configured
supabase_client = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        from supabase import create_client, Client
        supabase_client: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        log.info("Supabase client initialized successfully", url=SUPABASE_URL)
    except Exception as e:
        log.error("Failed to initialize Supabase client", error=str(e))
else:
    log.warning("Supabase environment variables not set. Running in LOCAL/MOCK mode.")


def is_supabase_active() -> bool:
    """Check if Supabase is active and configured."""
    return supabase_client is not None
