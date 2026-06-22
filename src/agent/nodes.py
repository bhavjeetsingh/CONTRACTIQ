"""
LangGraph Node Functions for ContractIQ
==========================================
Each function is a node in the contract processing graph.
Nodes read from ContractState and return partial state updates.
"""

import os
import sys
from typing import Dict, Any

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


def classify_request(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor node: classify what the user wants.

    Routes to appropriate pipeline based on request_type.
    If request_type is not set, defaults to "analyze".
    """
    request_type = state.get("request_type", "analyze")
    log.info("Request classified", request_type=request_type)
    return {"request_type": request_type}


def ingest_document(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Document ingestion node: read and extract text from uploaded file.

    Uses DocHandler which automatically falls back to OCR for scanned PDFs.
    """
    from src.document_ingestion.data_ingestion import DocHandler

    file_path = state.get("file_path", "")
    if not file_path:
        return {"error": "No file_path provided", "document_text": ""}

    try:
        dh = DocHandler()
        text = dh.read_document(file_path)

        # Check if OCR was used (indicated by minimal PyMuPDF text)
        is_ocr = False
        ocr_confidence = 1.0

        if file_path.lower().endswith(".pdf"):
            try:
                from src.ocr.ocr_pipeline import ContractOCR
                ocr = ContractOCR()
                is_ocr = ocr.is_scanned_pdf(file_path)
            except ImportError:
                pass

        log.info(
            "Document ingested",
            file_path=file_path,
            chars=len(text),
            is_ocr=is_ocr,
        )
        return {
            "document_text": text,
            "is_ocr": is_ocr,
            "ocr_confidence": ocr_confidence,
        }

    except Exception as e:
        log.error("Document ingestion failed", error=str(e))
        return {"error": f"Ingestion failed: {str(e)}", "document_text": ""}


def extract_key_terms(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Key-term extraction node: extract structured contract terms.

    Uses the KeyTermExtractor to pull obligations, payment terms,
    risk flags, etc. with confidence scores.
    """
    document_text = state.get("document_text", "")
    if not document_text:
        return {"error": "No document text available for extraction"}

    try:
        from src.document_analyzer.key_term_extractor import KeyTermExtractor

        extractor = KeyTermExtractor()
        result = extractor.extract(document_text)

        log.info(
            "Key terms extracted",
            fields=len(result.get("extracted_terms", {})),
            avg_confidence=result.get("avg_confidence", 0),
        )

        return {
            "extracted_terms": result.get("extracted_terms", {}),
            "confidence_scores": result.get("confidence_scores", {}),
        }

    except Exception as e:
        log.error("Key-term extraction failed", error=str(e))
        return {
            "error": f"Extraction failed: {str(e)}",
            "extracted_terms": {},
            "confidence_scores": {},
        }


def validate_extraction(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validation node: check extraction quality.

    If average confidence < 0.7, triggers retry.
    Checks for:
        - Missing critical fields (parties, effective_date)
        - Low confidence scores
        - Empty extractions
    """
    extracted = state.get("extracted_terms", {})
    scores = state.get("confidence_scores", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not extracted:
        log.warning("Validation failed: empty extraction")
        return {"validation_passed": False, "retry_count": retry_count}

    # Calculate average confidence
    if scores:
        avg_confidence = sum(scores.values()) / len(scores)
    else:
        avg_confidence = 0.5  # Default if no scores

    # Check critical fields
    critical_fields = ["parties", "effective_date", "governing_law"]
    missing_critical = [
        f for f in critical_fields
        if not extracted.get(f) or extracted.get(f) in ["", "Not found", "N/A"]
    ]

    # Validation logic
    passed = avg_confidence >= 0.7 and len(missing_critical) <= 1

    log.info(
        "Extraction validation",
        passed=passed,
        avg_confidence=round(avg_confidence, 3),
        missing_critical=missing_critical,
        retry_count=retry_count,
    )

    return {
        "validation_passed": passed,
        "retry_count": retry_count,
    }


def retry_extraction(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retry node: increment counter and re-route to extraction.

    Adds context about what was missing to improve the next attempt.
    """
    retry_count = state.get("retry_count", 0) + 1
    log.info("Retrying extraction", attempt=retry_count)
    return {"retry_count": retry_count}


def analyze_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Metadata analysis node: extract title, author, summary, sentiment.

    Uses the existing DocumentAnalyzer for backward compatibility.
    """
    document_text = state.get("document_text", "")
    if not document_text:
        return {"error": "No document text for analysis"}

    try:
        from src.document_analyzer.data_analysis import DocumentAnalyzer

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(document_text)

        log.info("Metadata analysis complete", keys=list(result.keys()))
        return {"analysis_result": result}

    except Exception as e:
        log.error("Metadata analysis failed", error=str(e))
        return {"error": f"Analysis failed: {str(e)}"}


def format_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Output formatting node: combine all results into final response.
    """
    output = {
        "analysis": state.get("analysis_result", {}),
        "key_terms": state.get("extracted_terms", {}),
        "confidence_scores": state.get("confidence_scores", {}),
        "ocr_info": {
            "is_ocr": state.get("is_ocr", False),
            "ocr_confidence": state.get("ocr_confidence", 1.0),
        },
        "validation": {
            "passed": state.get("validation_passed", True),
            "retries": state.get("retry_count", 0),
        },
        "method": state.get("method", "langgraph"),
    }

    if state.get("error"):
        output["error"] = state["error"]

    log.info("Output formatted", has_key_terms=bool(output["key_terms"]))
    return {"analysis_result": output, "method": "langgraph"}
