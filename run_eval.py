import os
import sys
import argparse
import random
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
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def run_eval(pdf_dir: str, dataset_path: str = None):
    print("Step 1: Loading 1 PDF document...")
    handler = DocHandler(session_id="eval_session_1_doc")
    
    class LocalFile:
        def __init__(self, path):
            self.path = path
            self.name = Path(path).name
        def read(self):
            return open(self.path, "rb").read()
        def getbuffer(self):
            return self.read()

    pdf_paths = list(set(Path(pdf_dir).glob("*.pdf")) | set(Path(pdf_dir).glob("*.PDF")))
    if not pdf_paths:
        print("No PDFs found.")
        return
        
    selected_pdf = random.choice(pdf_paths)
    print(f"   Selected PDF: {selected_pdf.name}")
    
    try:
        saved_path = handler.save_document(LocalFile(str(selected_pdf)))
        text = handler.read_document(saved_path)
        docs = [Document(page_content=text, metadata={"source": str(selected_pdf)})]
    except Exception as e:
        print(f"   Failed to load {selected_pdf.name}: {e}")
        return

    print("\n   Generating Question from this single PDF...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    prompt = f"Given the following document text, generate one specific question that can be answered from it, and provide the correct factual answer (ground truth). Format your output exactly like this:\nQuestion: <question>\nAnswer: <answer>\n\nText:\n{text[:6000]}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        q = ""
        a = ""
        for line in response.split("\n"):
            if line.startswith("Question:"):
                q = line.replace("Question:", "").strip()
            elif line.startswith("Answer:"):
                a = line.replace("Answer:", "").strip()
        if not q or not a:
            raise ValueError("Failed to parse QA format.")
        test_cases = [{"question": q, "ground_truth": a}]
        print(f"   Generated Q: {q}")
        print(f"   Generated A: {a}")
    except Exception as e:
        print(f"   Failed to generate question: {e}")
        return

    print("\nStep 2: Building Hybrid RAG pipeline for this 1 PDF...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)
    
    faiss_dir = "faiss_index/eval_session_1_doc"
    rag = ContractRAG(
        session_id="eval_session_1_doc",
        use_parent_retriever=False,
        k=3,
    )
    rag.build(docs, faiss_dir=faiss_dir)
    print("   Hybrid RAG built (BM25 + FAISS)")

    print("\nStep 3: Generating Answer & Scoring with Ragas...\n")
    print("="*60)
    
    try:
        results, scores = ContractEvalSuite.run_eval_with_rag(
            rag, 
            test_cases=test_cases, 
            save_results=True
        )
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", dest="pdf_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    run_eval(args.pdf_dir, dataset_path=args.dataset)
