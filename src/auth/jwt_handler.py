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

# --- Persistent user store (JSON file — replace with PostgreSQL for real production) ---
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
    try:
        import sys as _dbg
        print(f"[DEBUG] SECRET_KEY loaded: {bool(SECRET_KEY)} (len={len(SECRET_KEY) if SECRET_KEY else 0})", file=_dbg.stderr)
        print(f"[DEBUG] Token received: {token[:20]}... (len={len(token)})", file=_dbg.stderr)
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"[DEBUG] Payload: {payload}", file=_dbg.stderr)
        email: str = payload.get("sub")
        if email is None:
            print("[DEBUG] No 'sub' in payload", file=_dbg.stderr)
            raise credentials_exception
        print(f"[DEBUG] Auth OK for: {email}", file=_dbg.stderr)
        return TokenData(email=email)
    except JWTError as e:
        print(f"[DEBUG] JWTError: {e}", file=_dbg.stderr)
        raise credentials_exception

# --- Auth dependency (use this to protect routes) ---

def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    return decode_token(token)

# --- Auth operations ---

def register_user(email: str, password: str) -> dict:
    import re
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if email in _users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    _users_db[email] = hash_password(password)
    _save_users(_users_db)
    return {"message": "User registered successfully", "email": email}

def login_user(email: str, password: str) -> Token:
    hashed = _users_db.get(email)
    if not hashed or not verify_password(password, hashed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": email})
    return Token(access_token=token, token_type="bearer")
