"""
Run this script locally to test your RAG pipeline and generate QA pairs.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.documents import Document
from src.document_ingestion.data_ingestion import DocHandler
from src.document_chat.hybrid_retrieval import ContractRAG
from src.eval.ragas_evaluator import ContractEvalSuite
from langchain_text_splitters import RecursiveCharacterTextSplitter

def run_eval(pdf_path: str):
    """
    Run the standard test questions against a contract PDF.
    """

    print("Step 1: Loading document...")
    handler = DocHandler(session_id="eval_session")
    
    class LocalFile:
        def __init__(self, path):
            self.name = Path(path).name
        def read(self):
            return open(pdf_path, "rb").read()
        def getbuffer(self):
            return self.read()

    saved_path = handler.save_document(LocalFile(pdf_path))
    text = handler.read_document(saved_path)
    print(f"   Loaded {len(text)} characters")

    print("\nStep 2: Building Hybrid RAG pipeline...")
    docs = [Document(page_content=text, metadata={"source": pdf_path})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    faiss_dir = "faiss_index/eval_session"
    rag = ContractRAG(
        session_id="eval_session",
        use_parent_retriever=False,
        k=5,
    )
    rag.build(docs, faiss_dir=faiss_dir)
    print("   Hybrid RAG built (BM25 + FAISS)")

    print("\nStep 3: Generating Answers for Standard Test Cases...\n")
    print("="*60)
    
    try:
        results, scores = ContractEvalSuite.run_eval_with_rag(rag, save_results=True)
        
        for i, res in enumerate(results, 1):
            print(f"Q{i}: {res['question']}")
            print(f"Expected: {res['ground_truth']}")
            print(f"Answer: {res['answer']}")
            print("-" * 60)
        
        print("\n\n" + "="*20 + " RAGAS SCORES " + "="*20)
        for metric, score in scores.items():
            print(f"{metric.capitalize()}: {score}")
        print("="*54)

    except Exception as e:
        print(f"Evaluation Failed: {e}")

    print("\nEvaluation Q&A generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG pipeline on a contract PDF")
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

    run_eval(args.pdf)
