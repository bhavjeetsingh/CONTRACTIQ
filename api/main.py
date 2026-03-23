import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from constants import SUPPORTED_EXTENSIONS
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler
from logger import GLOBAL_LOGGER as log

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(
    title="Document Portal API", 
    version="0.2",
    description="Universal Document Processing API - Supports PDF, DOCX, PPT, Excel, CSV, TXT, JSON, RTF"
)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}

@app.get("/supported-formats")
def get_supported_formats() -> Dict[str, Any]:
    """Get list of supported file formats"""
    extensions = list(SUPPORTED_EXTENSIONS)
    return {
        "supported_extensions": extensions,
        "total_formats": len(extensions),
        "examples": {
            "documents": [".pdf", ".docx", ".txt", ".md", ".rtf"],
            "presentations": [".ppt", ".pptx"],  
            "spreadsheets": [".xlsx", ".xls", ".csv"],
            "data": [".json"]
        }
    }

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")
        
        # Validate file type
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            supported_formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Supported formats: {supported_formats}"
            )
        
        dh = DocHandler()
        saved_path = dh.save_document(FastAPIFileAdapter(file))  # Updated to use save_document
        text = dh.read_document(saved_path)  # Updated to use read_document
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable")
        
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        log.info("Document analysis complete.", file_type=file_ext, content_length=len(text))
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis", filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        
        # Validate file types
        ref_ext = Path(reference.filename or "").suffix.lower()
        act_ext = Path(actual.filename or "").suffix.lower()
        
        supported_formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        
        if ref_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Reference file type {ref_ext} not supported. Supported formats: {supported_formats}"
            )
        
        if act_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Actual file type {act_ext} not supported. Supported formats: {supported_formats}"
            )
        
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        
        combined_text = dc.combine_documents()
        
        if not combined_text or not combined_text.strip():
            raise HTTPException(status_code=400, detail="One or both documents appear to be empty or unreadable")
        
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        log.info("Document comparison completed.", 
                ref_type=ref_ext, 
                act_type=act_ext,
                content_length=len(combined_text))
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed", 
                     ref_file=reference.filename, 
                     act_file=actual.filename)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        
        # Validate all files before processing
        unsupported_files = []
        supported_formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        
        for file in files:
            file_ext = Path(file.filename or "").suffix.lower()
            if file_ext not in SUPPORTED_EXTENSIONS:
                unsupported_files.append(f"{file.filename} ({file_ext})")
        
        if unsupported_files:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file types found: {'; '.join(unsupported_files)}. Supported formats: {supported_formats}"
            )
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided for indexing")
        
        wrapped = [FastAPIFileAdapter(f) for f in files]
        
        # Create ChatIngestor instance
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        
        # Build retriever (note: method name might need fixing if it's actually build_retriever)
        retriever = ci.built_retriver(
            wrapped, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            k=k
        )
        
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {
            "session_id": ci.session_id, 
            "k": k, 
            "use_session_dirs": use_session_dirs,
            "files_indexed": len(files),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed", 
                     files=[f.filename for f in files])
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.", 
                question_length=len(question),
                response_length=len(str(response)))

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG",
            "question": question
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed", question=question, session=session_id)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# ---------- UTILITY ENDPOINTS ----------

@app.get("/sessions")
async def list_sessions() -> Dict[str, Any]:
    """List available chat sessions"""
    try:
        sessions = []
        faiss_base_path = Path(FAISS_BASE)
        
        if faiss_base_path.exists():
            for session_dir in faiss_base_path.iterdir():
                if session_dir.is_dir() and (session_dir / "index.faiss").exists():
                    sessions.append({
                        "session_id": session_dir.name,
                        "path": str(session_dir),
                        "created": session_dir.stat().st_ctime
                    })
        
        sessions.sort(key=lambda x: x["created"], reverse=True)
        return {"sessions": sessions, "total": len(sessions)}
        
    except Exception as e:
        log.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a specific chat session"""
    try:
        import shutil
        session_path = Path(FAISS_BASE) / session_id
        
        if not session_path.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        shutil.rmtree(session_path)
        log.info("Session deleted", session_id=session_id)
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to delete session", session_id=session_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

# command for executing the FastAPI app
# uvicorn api.main:app --port 8080 --reload    
# uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload