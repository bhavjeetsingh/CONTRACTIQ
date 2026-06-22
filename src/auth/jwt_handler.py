import os
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from logger import GLOBAL_LOGGER as log
from src.database.supabase_client import is_supabase_active, supabase_client
from src.database.supabase_db import create_profile_if_not_exists

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    import sys as _sys
    print("[WARNING] SECRET_KEY not set — generated ephemeral key. Tokens will NOT survive restarts. Set SECRET_KEY env var for production.", file=_sys.stderr)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Use pbkdf2_sha256 for stable cross-platform hashing without bcrypt backend issues.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# --- Schemas ---

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    id: Optional[str] = None

# --- Persistent user store (JSON file — fallback when Supabase is disabled) ---
_USERS_FILE = Path(os.getenv("USERS_DB_PATH", "data/users.json"))

def _load_users() -> dict:
    if _USERS_FILE.exists():
        try:
            return json.loads(_USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_users(db: dict):
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USERS_FILE.write_text(json.dumps(db, indent=2), encoding="utf-8")

_users_db: dict = _load_users()

# --- Password helpers ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# --- Token helpers ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if is_supabase_active():
        # Local JWT Decode if secret is provided (fast and serverless-friendly)
        from src.database.supabase_client import SUPABASE_JWT_SECRET
        if SUPABASE_JWT_SECRET:
            try:
                payload = jwt.decode(
                    token, 
                    SUPABASE_JWT_SECRET, 
                    algorithms=["HS256"], 
                    audience="authenticated"
                )
                email = payload.get("email")
                user_id = payload.get("sub") # 'sub' is the UUID in Supabase JWTs
                if email is None or user_id is None:
                    raise credentials_exception
                return TokenData(email=email, id=user_id)
            except JWTError as e:
                log.warning("Local Supabase JWT decode failed, falling back to API verification", error=str(e))
        
        # Fallback: verification via Supabase API
        try:
            res = supabase_client.auth.get_user(token)
            if not res or not res.user:
                raise credentials_exception
            return TokenData(email=res.user.email, id=res.user.id)
        except Exception as e:
            log.error("Supabase JWT verification failed", error=str(e))
            raise credentials_exception
    else:
        # Legacy local JWT validation
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                raise credentials_exception
            # Generate a deterministic user UUID from email in mock mode
            import hashlib
            import uuid
            user_id = str(uuid.UUID(hashlib.md5(email.encode()).hexdigest()))
            return TokenData(email=email, id=user_id)
        except JWTError:
            raise credentials_exception

# --- Auth dependency (use this to protect routes) ---

def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    user_data = decode_token(token)
    # Ensure a profile row exists in local DB or Supabase profiles table
    if user_data.id and user_data.email:
        create_profile_if_not_exists(user_data.id, user_data.email)
    return user_data

# --- Auth operations ---

def register_user(email: str, password: str) -> dict:
    import re
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
    if is_supabase_active():
        try:
            res = supabase_client.auth.sign_up({"email": email, "password": password})
            if not res.user:
                raise HTTPException(status_code=400, detail="Registration failed")
            create_profile_if_not_exists(res.user.id, email)
            return {"message": "User registered successfully", "email": email, "id": res.user.id}
        except Exception as e:
            log.error("Supabase sign up error", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
    else:
        if email in _users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        _users_db[email] = hash_password(password)
        _save_users(_users_db)
        return {"message": "User registered successfully", "email": email}

def login_user(email: str, password: str) -> Token:
    if is_supabase_active():
        try:
            res = supabase_client.auth.sign_in_with_password({"email": email, "password": password})
            if not res.session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            create_profile_if_not_exists(res.user.id, email)
            return Token(access_token=res.session.access_token, token_type="bearer")
        except Exception as e:
            log.error("Supabase sign in error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        hashed = _users_db.get(email)
        if not hashed or not verify_password(password, hashed):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = create_access_token(data={"sub": email})
        return Token(access_token=token, token_type="bearer")
