"""
Run this script locally to generate RAGAS eval scores for your README.

Usage:
    python run_eval.py --pdf path/to/contract.pdf

This will:
1. Load the PDF
2. Build a ContractRAG instance
3. Run 5 standard contract questions through it
4. Score faithfulness, answer relevancy, context precision, context recall
5. Save results to eval_results/eval_TIMESTAMP.json
6. Print scores to copy into your README

Requirements:
    pip install ragas datasets
    GROQ_API_KEY or GOOGLE_API_KEY must be set in .env
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.document_ingestion.data_ingestion import DocHandler, ChatIngestor
from src.document_chat.hybrid_retrieval import ContractRAG
from src.eval.ragas_evaluator import ContractEvalSuite
from utils.document_ops import load_documents
from logger import GLOBAL_LOGGER as log


def run_eval(pdf_path: str):
    """
    Run full RAGAS evaluation on a contract PDF.

    Args:
        pdf_path: Path to a contract PDF file
    """
    print(f"\n ContractIQ — RAGAS Evaluation")
    print(f"{'='*50}")
    print(f"Document: {pdf_path}")
    print(f"{'='*50}\n")

    # Step 1: Load and index document
    print("Step 1: Loading document...")
    handler = DocHandler(session_id="eval_session")
    
    class LocalFile:
        def __init__(self, path):
            self.name = Path(path).name
        def getbuffer(self):
            return open(pdf_path, "rb").read()

    saved_path = handler.save_document(LocalFile(pdf_path))
    text = handler.read_document(saved_path)
    print(f"   Loaded {len(text)} characters")

    # Step 2: Build RAG pipeline
    print("\nStep 2: Building Hybrid RAG pipeline...")
    from langchain.schema import Document
    docs = [Document(page_content=text, metadata={"source": pdf_path})]

    faiss_dir = "faiss_index/eval_session"
    rag = ContractRAG(
        session_id="eval_session",
        use_parent_retriever=False,
        k=5,
    )
    rag.build(docs, faiss_dir=faiss_dir)
    print("   Hybrid RAG built (BM25 + FAISS)")

    # Step 3: Run RAGAS evaluation
    print("\nStep 3: Running RAGAS evaluation (this takes 1-2 minutes)...")
    scores = ContractEvalSuite.run_eval_with_rag(rag, save_results=True)

    # Step 4: Print results
    print(f"\n{'='*50}")
    print(" RAGAS Evaluation Results")
    print(f"{'='*50}")
    print(f"  Faithfulness:      {scores['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.4f}")
    print(f"  Context Precision: {scores['context_precision']:.4f}")
    print(f"  Context Recall:    {scores['context_recall']:.4f}")
    print(f"  Test Cases:        {scores['test_cases']}")
    print(f"{'='*50}")

    print("\n README badge format:")
    print(f"  Faithfulness: {scores['faithfulness']:.2f} | ", end="")
    print(f"Answer Relevancy: {scores['answer_relevancy']:.2f} | ", end="")
    print(f"Context Precision: {scores['context_precision']:.2f}")

    print(f"\n Results saved to eval_results/")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on a contract PDF")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to contract PDF file",
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"Error: File not found: {args.pdf}")
        sys.exit(1)

    scores = run_eval(args.pdf)
