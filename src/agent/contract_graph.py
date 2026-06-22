"""
ContractGraph — LangGraph StateGraph for ContractIQ
======================================================
Stateful, agentic workflow that replaces the linear LCEL chain.

Graph structure:
    classify -> ingest -> extract -> validate -> [retry or format]

Features:
    - Automatic routing based on request type
    - Self-correcting extraction with retry loop (max 2 retries)
    - Per-field confidence scoring and validation
    - Backward-compatible with existing analyze/chat endpoints
"""

from typing import Dict, Any, Optional

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    log.warning("LangGraph not installed. Install: pip install langgraph")

from src.agent.state import ContractState
from src.agent.nodes import (
    classify_request,
    ingest_document,
    extract_key_terms,
    validate_extraction,
    retry_extraction,
    analyze_metadata,
    format_output,
)


def _should_retry(state: Dict[str, Any]) -> str:
    """
    Conditional edge: decide whether to retry extraction or proceed.

    Returns:
        "retry" if validation failed and retries remain
        "format" if validation passed or max retries reached
    """
    if state.get("validation_passed", False):
        return "format"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        log.info(
            "Routing to retry",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        return "retry"
    else:
        log.warning(
            "Max retries reached, proceeding with best result",
            retry_count=retry_count,
        )
        return "format"


def _route_request(state: Dict[str, Any]) -> str:
    """
    Conditional edge: route based on request type.

    Returns node name to route to.
    """
    request_type = state.get("request_type", "analyze")

    if request_type in ("analyze", "extract"):
        return "ingest"
    else:
        return "ingest"  # Default: always ingest first


def build_contract_graph() -> Optional[Any]:
    """
    Build the LangGraph StateGraph for contract processing.

    Returns:
        Compiled LangGraph graph, or None if LangGraph is not available.

    Graph topology:
        classify -> ingest -> analyze_metadata -> extract -> validate
                                                              |
                                                    [retry] <-+-> [format] -> END
    """
    if not LANGGRAPH_AVAILABLE:
        log.warning("Cannot build contract graph: LangGraph not installed")
        return None

    try:
        graph = StateGraph(ContractState)

        # Add nodes
        graph.add_node("classify", classify_request)
        graph.add_node("ingest", ingest_document)
        graph.add_node("analyze_metadata", analyze_metadata)
        graph.add_node("extract", extract_key_terms)
        graph.add_node("validate", validate_extraction)
        graph.add_node("retry", retry_extraction)
        graph.add_node("format", format_output)

        # Set entry point
        graph.set_entry_point("classify")

        # Add edges
        graph.add_conditional_edges("classify", _route_request, {
            "ingest": "ingest",
        })
        graph.add_edge("ingest", "analyze_metadata")
        graph.add_edge("analyze_metadata", "extract")
        graph.add_edge("extract", "validate")
        graph.add_conditional_edges("validate", _should_retry, {
            "retry": "retry",
            "format": "format",
        })
        graph.add_edge("retry", "extract")
        graph.add_edge("format", END)

        compiled = graph.compile()
        log.info("Contract graph compiled successfully")
        return compiled

    except Exception as e:
        log.error("Failed to build contract graph", error=str(e))
        raise DocumentPortalException("Failed to build contract graph", e) from e


# Singleton compiled graph
_contract_graph = None


def get_contract_graph():
    """Get or create the singleton contract graph."""
    global _contract_graph
    if _contract_graph is None:
        _contract_graph = build_contract_graph()
    return _contract_graph


def run_contract_pipeline(
    file_path: str,
    request_type: str = "analyze",
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Run the full contract processing pipeline.

    This is the main entry point for the agentic workflow.
    Falls back to the legacy linear chain if LangGraph is not available.

    Args:
        file_path: Path to the uploaded document
        request_type: "analyze" or "extract"
        max_retries: Max extraction retry attempts

    Returns:
        Dict with analysis results, key terms, and confidence scores
    """
    graph = get_contract_graph()

    if graph is None:
        # Fallback to legacy linear pipeline
        log.warning("LangGraph not available, using legacy pipeline")
        from src.document_ingestion.data_ingestion import DocHandler
        from src.document_analyzer.data_analysis import DocumentAnalyzer

        dh = DocHandler()
        text = dh.read_document(file_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        return {"analysis": result, "method": "legacy_linear"}

    # Run the graph
    initial_state = {
        "request_type": request_type,
        "file_path": file_path,
        "file_type": file_path.rsplit(".", 1)[-1] if "." in file_path else "",
        "document_text": "",
        "is_ocr": False,
        "ocr_confidence": 1.0,
        "query": "",
        "chat_history": [],
        "retrieved_docs": [],
        "extracted_terms": {},
        "confidence_scores": {},
        "validation_passed": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "analysis_result": {},
        "error": None,
        "method": "langgraph",
    }

    log.info("Running contract pipeline", file_path=file_path, request_type=request_type)
    final_state = graph.invoke(initial_state)

    result = final_state.get("analysis_result", {})
    if final_state.get("error"):
        result["error"] = final_state["error"]

    log.info(
        "Contract pipeline complete",
        method=final_state.get("method", "unknown"),
        retries=final_state.get("retry_count", 0),
        validation=final_state.get("validation_passed", False),
    )

    return result
