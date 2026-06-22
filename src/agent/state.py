"""
ContractState — Shared state for the LangGraph contract pipeline.
Every node reads from and writes to this state.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class ContractState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph contract pipeline.

    Fields:
        request_type: What the user wants — "analyze", "extract", "chat", "compare"
        document_text: Raw text extracted from the uploaded document
        file_path: Path to the uploaded document
        file_type: File extension (e.g., ".pdf", ".docx")
        is_ocr: Whether OCR was used for text extraction
        ocr_confidence: Average OCR confidence score (1.0 for digital PDFs)

        query: User's question (for chat mode)
        chat_history: Previous conversation turns

        retrieved_docs: Documents retrieved by hybrid search
        extracted_terms: Structured key-term extraction result
        confidence_scores: Per-field confidence scores
        
        validation_passed: Whether the extraction passed quality validation
        retry_count: Number of extraction retry attempts
        max_retries: Maximum allowed retries

        analysis_result: Final analysis output (metadata, summary, etc.)
        error: Error message if something failed
        method: Which extraction method was used
    """
    # Input
    request_type: str
    document_text: str
    file_path: str
    file_type: str

    # OCR metadata
    is_ocr: bool
    ocr_confidence: float

    # Chat context
    query: str
    chat_history: list

    # Retrieval
    retrieved_docs: List[Document]

    # Extraction
    extracted_terms: Dict[str, Any]
    confidence_scores: Dict[str, float]

    # Validation & retry
    validation_passed: bool
    retry_count: int
    max_retries: int

    # Output
    analysis_result: Dict[str, Any]
    error: Optional[str]
    method: str
