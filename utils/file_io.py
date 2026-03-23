from __future__ import annotations
import re
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Iterable, List, Union
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

# Import from the main data_ingestion module to maintain consistency
from src.document_ingestion.data_ingestion import SUPPORTED_EXTENSIONS

# ----------------------------- #
# Helpers (file I/O + loading)  #
# ----------------------------- #

def generate_session_id(prefix: str = "session") -> str:
    """Generate a unique session ID with timestamp and random component"""
    try:
        ist = ZoneInfo("Asia/Kolkata")
        timestamp = datetime.now(ist).strftime('%Y%m%d_%H%M%S')
        random_suffix = uuid.uuid4().hex[:8]
        session_id = f"{prefix}_{timestamp}_{random_suffix}"
        log.info("Generated session ID", session_id=session_id)
        return session_id
    except Exception as e:
        # Fallback to simpler generation if timezone fails
        log.warning("Fallback session ID generation", error=str(e))
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """
    Save uploaded files and return local paths.
    Now supports all document types defined in SUPPORTED_EXTENSIONS
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        
        for uf in uploaded_files:
            try:
                # Get file name - handle different input types
                name = getattr(uf, "name", None) or getattr(uf, "filename", "unknown_file")
                
                # Validate file extension
                ext = Path(name).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    log.warning("Unsupported file skipped", 
                               filename=name, 
                               extension=ext,
                               supported=list(SUPPORTED_EXTENSIONS))
                    continue
                
                # Generate safe filename
                safe_name = _generate_safe_filename(name, ext)
                out_path = target_dir / safe_name
                
                # Save file content
                with open(out_path, "wb") as f:
                    if hasattr(uf, "read"):
                        f.write(uf.read())
                    elif hasattr(uf, "getbuffer"):
                        f.write(uf.getbuffer())
                    else:
                        raise ValueError(f"Cannot read file content from {type(uf)}")
                
                saved.append(out_path)
                log.info("File saved for ingestion", 
                        original=name, 
                        saved_as=str(out_path),
                        size=out_path.stat().st_size)
                        
            except Exception as file_error:
                log.error("Failed to save individual file", 
                         filename=getattr(uf, "name", "unknown"),
                         error=str(file_error))
                # Continue with other files instead of failing completely
                continue
        
        if not saved:
            log.warning("No files were saved", 
                       total_attempted=len(list(uploaded_files)))
        
        return saved
        
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e

def _generate_safe_filename(original_name: str, extension: str) -> str:
    """
    Generate a safe filename from the original name
    """
    try:
        # Extract the base name without extension
        base_name = Path(original_name).stem
        
        # Clean the filename - keep only alphanumeric, dash, underscore
        safe_base = re.sub(r'[^a-zA-Z0-9_\-]', '_', base_name)
        
        # Ensure it's not empty and not too long
        if not safe_base or len(safe_base) < 2:
            safe_base = "document"
        elif len(safe_base) > 50:
            safe_base = safe_base[:50]
        
        # Add unique identifier to prevent conflicts
        unique_id = uuid.uuid4().hex[:8]
        
        # Combine everything
        safe_name = f"{safe_base}_{unique_id}{extension}"
        
        return safe_name.lower()
        
    except Exception as e:
        log.warning("Error generating safe filename, using fallback", 
                   original=original_name, error=str(e))
        # Fallback to completely generic name
        return f"document_{uuid.uuid4().hex[:8]}{extension}"

def validate_file_type(filename: str) -> bool:
    """
    Validate if a file type is supported
    """
    if not filename:
        return False
        
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS

def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get basic information about a file
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File does not exist", "exists": False}
        
        stat = path.stat()
        
        return {
            "exists": True,
            "name": path.name,
            "stem": path.stem,
            "extension": path.suffix.lower(),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "is_supported": path.suffix.lower() in SUPPORTED_EXTENSIONS,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "full_path": str(path.absolute())
        }
        
    except Exception as e:
        return {"error": str(e), "exists": False}

def cleanup_old_files(directory: Path, max_age_hours: int = 24, keep_count: int = 10):
    """
    Clean up old files in a directory
    Keep the most recent 'keep_count' files and delete files older than 'max_age_hours'
    """
    try:
        if not directory.exists():
            return
            
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Get all files sorted by modification time (newest first)
        files = []
        for file_path in directory.iterdir():
            if file_path.is_file():
                files.append((file_path, file_path.stat().st_mtime))
        
        files.sort(key=lambda x: x[1], reverse=True)
        
        deleted_count = 0
        
        # Keep the newest 'keep_count' files, delete the rest
        for i, (file_path, mtime) in enumerate(files):
            file_time = datetime.fromtimestamp(mtime)
            
            # Delete if beyond keep count OR if older than cutoff
            if i >= keep_count or file_time < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    log.info("Old file deleted", file=str(file_path), age_hours=(datetime.now() - file_time).total_seconds() / 3600)
                except Exception as del_error:
                    log.error("Failed to delete old file", file=str(file_path), error=str(del_error))
        
        if deleted_count > 0:
            log.info("File cleanup completed", directory=str(directory), deleted=deleted_count, kept=len(files) - deleted_count)
            
    except Exception as e:
        log.error("Error during file cleanup", directory=str(directory), error=str(e))

def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create it if it doesn't
    """
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        log.error("Failed to create directory", directory=str(directory), error=str(e))
        raise DocumentPortalException(f"Failed to create directory: {directory}", e) from e

def get_supported_extensions_info() -> dict:
    """
    Get information about supported file extensions
    """
    return {
        "extensions": list(SUPPORTED_EXTENSIONS),
        "count": len(SUPPORTED_EXTENSIONS),
        "by_category": {
            "documents": [ext for ext in SUPPORTED_EXTENSIONS if ext in {".pdf", ".docx", ".txt", ".md", ".rtf"}],
            "presentations": [ext for ext in SUPPORTED_EXTENSIONS if ext in {".ppt", ".pptx"}],
            "spreadsheets": [ext for ext in SUPPORTED_EXTENSIONS if ext in {".xlsx", ".xls", ".csv"}],
            "data": [ext for ext in SUPPORTED_EXTENSIONS if ext in {".json"}]
        }
    }