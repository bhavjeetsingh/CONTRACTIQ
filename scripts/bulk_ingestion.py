import os
import sys
import argparse
from pathlib import Path

# Ensure the root directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import PyMuPDFLoader
from src.document_chat.hybrid_retrieval import HybridRetriever12
from logger import GLOBAL_LOGGER as log

def bulk_ingest(input_dir: str, session_id: str):
    """
    Offline job to bulk ingest a directory of PDFs into a FAISS index.
    """
    folder_path = Path(input_dir)
    if not folder_path.exists() or not folder_path.is_dir():
        log.error(f"Directory not found: {folder_path}")
        return

    log.info(f"Scanning {input_dir} for PDFs...")
    documents = []
    
    # Extract text from all PDFs
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())

    log.info(f"Successfully loaded {len(documents)} pages from PDFs.")
    log.info("Initializing Hybrid Retriever & running local BGE embeddings...")

    # Initialize Retriever & Build FAISS Index
    retriever = HybridRetriever()
    faiss_path = os.path.join("data", "faiss_index", session_id)
    retriever.build(documents, faiss_index_dir=faiss_path)

    log.info(f"SUCCESS! FAISS Index '{session_id}' created securely at {faiss_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Ingest PDFs into FAISS")
    parser.add_argument("--dir", type=str, required=True, help="Path to the folder containing PDFs")
    parser.add_argument("--session", type=str, default="bulk_corpus_01", help="Name/Session ID for the FAISS index")
    
    args = parser.parse_args()
    bulk_ingest(args.dir, args.session)
