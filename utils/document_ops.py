"""
Universal document operations utility
Supports PDF, DOCX, PPT, Excel, CSV, TXT, JSON, RTF files
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Optional, Union, Any, Iterable
from fastapi import UploadFile
from langchain.schema import Document
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

# Import constants to avoid circular imports
from constants import SUPPORTED_EXTENSIONS

class FastAPIFileAdapter:
    """
    Adapter to make FastAPI UploadFile compatible with our document processing
    Compatible with both old and new interfaces
    """
    def __init__(self, upload_file: UploadFile):
        self.upload_file = upload_file
        self.name = upload_file.filename or "unknown"

    def read(self) -> bytes:
        """Read file content as bytes"""
        self.upload_file.file.seek(0)  # Ensure we start from beginning
        content = self.upload_file.file.read()
        self.upload_file.file.seek(0)  # Reset file pointer for potential re-reads
        return content

    def getbuffer(self) -> bytes:
        """Alternative method for getting file buffer - maintains backward compatibility"""
        return self.read()

def load_documents(file_paths: Union[List[Union[str, Path]], Iterable[Path]]) -> List[Document]:
    """
    Load documents from file paths into LangChain Document objects
    Supports all file types defined in SUPPORTED_EXTENSIONS
    Compatible with both new paths list and old Iterable[Path] interface
    """
    # Import here to avoid circular imports
    from src.document_ingestion.data_ingestion import DocHandler
    
    documents = []
    doc_handler = DocHandler()
    
    # Convert to list of Path objects for consistent handling
    if isinstance(file_paths, (list, tuple)):
        paths = [Path(p) for p in file_paths]
    else:
        paths = list(file_paths)  # Handle Iterable[Path]
    
    for file_path in paths:
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                log.warning("File not found", path=str(file_path))
                continue
                
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file type", path=str(file_path), extension=file_path.suffix)
                continue
            
            # Read document content using our universal reader
            content = doc_handler.read_document(str(file_path))
            
            if content and content.strip():
                # Create LangChain Document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file_path": str(file_path),  # For backward compatibility
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "file_size": file_path.stat().st_size if file_path.exists() else 0
                    }
                )
                documents.append(doc)
                log.info("Document loaded successfully", 
                        file=file_path.name, 
                        type=file_path.suffix, 
                        content_length=len(content))
            else:
                log.warning("Empty document content", path=str(file_path))
                
        except Exception as e:
            log.error("Failed to load document", 
                     path=str(file_path), 
                     error=str(e))
            # Continue with other files instead of failing completely
            continue
    
    log.info("Document loading completed", 
             total_files=len(paths), 
             loaded=len(documents))
    return documents

def read_pdf_via_handler(doc_handler, pdf_path: str) -> str:
    """
    Legacy function for backward compatibility
    Now uses the universal read_document method
    """
    return doc_handler.read_document(pdf_path)

def concat_for_analysis(documents: List[Document]) -> str:
    """
    Concatenate documents for analysis
    Maintains document boundaries and metadata
    Compatible with both old and new document structures
    """
    try:
        if not documents:
            return ""
            
        content_parts = []
        
        for i, doc in enumerate(documents):
            # Handle both old and new metadata formats
            file_name = (doc.metadata.get('file_name') or 
                        doc.metadata.get('source', '').split('/')[-1] or 
                        f'Document_{i+1}')
            file_type = doc.metadata.get('file_type', 'unknown')
            source = doc.metadata.get('source') or doc.metadata.get('file_path') or 'unknown'
            
            # Create header that matches old format for compatibility
            header = f"\n--- SOURCE: {source} ---\n"
            content_parts.append(header + doc.page_content)
        
        combined = "\n".join(content_parts)
        log.info("Documents concatenated for analysis", 
                count=len(documents), 
                total_length=len(combined))
        return combined
        
    except Exception as e:
        log.error("Error concatenating documents for analysis", error=str(e))
        raise DocumentPortalException("Failed to concatenate documents for analysis", e) from e

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    """
    Concatenate documents for comparison - maintains backward compatibility
    """
    try:
        left = concat_for_analysis(ref_docs)
        right = concat_for_analysis(act_docs)
        return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"
        
    except Exception as e:
        log.error("Error concatenating documents for comparison", error=str(e))
        raise DocumentPortalException("Failed to concatenate documents for comparison", e) from e

def extract_document_metadata(file_path: Union[str, Path]) -> dict:
    """
    Extract metadata from any supported document type
    """
    try:
        file_path = Path(file_path)
        
        basic_metadata = {
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "file_path": str(file_path)
        }
        
        # Add file-type specific metadata if needed
        if file_path.suffix.lower() == '.pdf':
            try:
                import fitz
                with fitz.open(file_path) as doc:
                    basic_metadata.update({
                        "page_count": doc.page_count,
                        "pdf_metadata": doc.metadata
                    })
            except Exception:
                pass
                
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                import pandas as pd
                excel_file = pd.ExcelFile(file_path)
                basic_metadata.update({
                    "sheet_names": excel_file.sheet_names,
                    "sheet_count": len(excel_file.sheet_names)
                })
            except Exception:
                pass
        
        return basic_metadata
        
    except Exception as e:
        log.error("Error extracting document metadata", 
                 file=str(file_path), 
                 error=str(e))
        return {"error": str(e)}

def validate_document_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if a file is supported and readable
    """
    try:
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return False
            
        # Check if extension is supported
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
            
        # Check if file is readable (basic check)
        if file_path.stat().st_size == 0:
            return False
            
        return True
        
    except Exception:
        return False

def get_supported_extensions() -> set:
    """
    Get list of supported file extensions
    """
    return SUPPORTED_EXTENSIONS.copy()

def format_document_for_display(doc: Document, max_length: int = 500) -> dict:
    """
    Format document for display in UI
    """
    try:
        content = doc.page_content
        
        # Truncate content if too long
        if len(content) > max_length:
            content = content[:max_length] + "..."
            
        return {
            "content": content,
            "metadata": doc.metadata,
            "content_length": len(doc.page_content),
            "truncated": len(doc.page_content) > max_length
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "content": "",
            "metadata": {},
            "content_length": 0,
            "truncated": False
        }