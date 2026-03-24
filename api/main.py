import os
from src.document_chat.hybrid_retrieval import ContractRAG
from src.cache.redis_cache import cache
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from utils.rate_limiter import limiter, get_user_identifier
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.document_chat.sse_streaming import router as stream_router, stream_rag_response
from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.eval.ragas_evaluator import RAGASEvaluator, ContractEvalSuite
from constants import SUPPORTED_EXTENSIONS
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler
from logger import GLOBAL_LOGGER as log
from src.auth.jwt_handler import (
    UserCreate, Token, register_user, 
    login_user, get_current_user, TokenData
)

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(
    title="ContractIQ API", 
    version="0.2",
    description="Universal Document Processing API - Supports PDF, DOCX, PPT, Excel, CSV, TXT, JSON, RTF"
)
app.include_router(stream_router)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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

@app.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    log.info("Serving login page.")
    resp = templates.TemplateResponse("login.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}
@app.post("/auth/register")
def register(user: UserCreate):
    return register_user(user.email, user.password)

@app.post("/auth/login", response_model=Token)
def login(user: UserCreate):
    return login_user(user.email, user.password)

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
@limiter.limit("10/minute")
@app.post("/analyze")
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user)
) -> Any:
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
@limiter.limit("10/minute")
@app.post("/compare")
async def compare_documents(
    request: Request,
    reference: UploadFile = File(...),
    actual: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user)
) -> Any:
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

# ---------- CHAT: INDEX (Hybrid RAG) ----------
@limiter.limit("5/minute")
@app.post("/chat/index")
async def chat_build_index(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
    use_parent_retriever: bool = Form(False),
    current_user: TokenData = Depends(get_current_user)
) -> Any:
    try:
        log.info(f"Indexing session. Files: {[f.filename for f in files]}")
        wrapped = [FastAPIFileAdapter(f) for f in files]
 
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
 
        paths = ci.save_files(wrapped)
        docs = ci.load_documents(paths)
 
        faiss_dir = os.path.join(FAISS_BASE, ci.session_id) if use_session_dirs else FAISS_BASE
 
        rag = ContractRAG(
            session_id=ci.session_id,
            use_parent_retriever=use_parent_retriever,
            k=k,
        )
        rag.build(docs, faiss_dir=faiss_dir)
 
        # Store in app state (in-memory)
        app.state.rag_sessions = getattr(app.state, "rag_sessions", {})
        app.state.rag_sessions[ci.session_id] = rag
 
        # Store session metadata in Redis (survives restarts)
        cache.store_rag_session(ci.session_id, {
            "faiss_dir": faiss_dir,
            "k": k,
            "use_parent_retriever": use_parent_retriever,
            "files": [f.filename for f in files],
            "user": current_user.email,
        })
 
        # Invalidate any old cached responses for this session
        cache.invalidate_session(ci.session_id)
 
        log.info(f"Hybrid RAG index built for session: {ci.session_id}")
        return {
            "session_id": ci.session_id,
            "k": k,
            "retriever_type": "parent" if use_parent_retriever else "hybrid",
            "files_indexed": len(files),
            "cache_status": "active" if cache.is_available else "disabled",
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
# ---------- CHAT: QUERY (Hybrid RAG) ----------
@limiter.limit("20/minute")
@app.post("/chat/query")
async def chat_query(
    request: Request,
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    current_user: TokenData = Depends(get_current_user)
) -> Any:
    try:
        log.info(f"Chat query: '{question}' | session: {session_id}")
 
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id required")
 
        # --- Check cache first ---
        cached = cache.get(session_id, question, "hybrid")
        if cached:
            log.info("Returning cached response", session_id=session_id)
            cached["from_cache"] = True
            return cached
 
        # --- Get RAG from app state ---
        rag_sessions = getattr(app.state, "rag_sessions", {})
        rag = rag_sessions.get(session_id)
 
        # --- Rebuild from Redis session metadata if app restarted ---
        if rag is None:
            session_meta = cache.get_rag_session(session_id)
 
            if session_meta:
                log.info(f"Rebuilding RAG from Redis session metadata: {session_id}")
                rag = ContractRAG(
                    session_id=session_id,
                    use_parent_retriever=session_meta.get("use_parent_retriever", False),
                    k=session_meta.get("k", k),
                )
                faiss_dir = session_meta.get("faiss_dir")
                if faiss_dir and os.path.isdir(faiss_dir):
                    # Rebuild from saved FAISS index
                    from langchain_community.vectorstores import FAISS
                    from utils.model_loader import ModelLoader
                    loader = ModelLoader()
                    embeddings = loader.load_embeddings()
                    vs = FAISS.load_local(
                        faiss_dir,
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    # Use basic retriever as fallback when rebuilding
                    rag_fallback = ConversationalRAG(session_id=session_id)
                    rag_fallback.load_retriever_from_faiss(
                        faiss_dir, k=k, index_name=FAISS_INDEX_NAME
                    )
                    response = rag_fallback.invoke(question, chat_history=[])
                    result = {
                        "answer": response,
                        "session_id": session_id,
                        "retriever_type": "basic-rebuild",
                        "from_cache": False,
                    }
                    # Cache the rebuilt response
                    cache.set(result, session_id, question, "hybrid")
                    return result
 
            # No session found at all
            index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
            if not os.path.isdir(index_dir):
                raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
 
        # --- Invoke hybrid RAG ---
        result = rag.invoke(question, chat_history=[])
        result["from_cache"] = False
 
        # --- Store response in cache ---
        cache.set(result, session_id, question, "hybrid")
 
        log.info("Hybrid RAG query handled successfully")
        return result
 
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
 
    
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

# cache stats for monitoring
@app.get("/cache/stats")
def cache_stats(current_user: TokenData = Depends(get_current_user)):
    """Get Redis cache statistics — useful for monitoring in production."""
    return cache.get_stats()


# This is the streaming endpoint
@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    k: int = Form(5),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Stream LLM response token by token via Server-Sent Events.

    Client usage (JavaScript):
        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Authorization': 'Bearer <token>' },
            body: new FormData({ question, session_id })
        });

        const reader = response.body.getReader();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = new TextDecoder().decode(value);
            const lines = text.split('\\n\\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.token) displayToken(data.token);
                    if (data.done) console.log('Stream complete');
                }
            }
        }
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    log.info("SSE stream started", session_id=session_id, question=question[:80])

    return StreamingResponse(
        stream_rag_response(
            question=question,
            session_id=session_id,
            app_state=request.app.state,
            k=k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
# evaluation endpoint of ragas
@app.post("/eval/score")
@limiter.limit("3/minute")
async def evaluate_rag(
    request: Request,
    session_id: str = Form(...),
    current_user: TokenData = Depends(get_current_user),
) -> Any:
    """
    Run RAGAS evaluation on an existing RAG session.
    Returns faithfulness, answer relevancy, context precision scores.
    """
    try:
        log.info("Starting RAGAS evaluation", session_id=session_id)
 
        rag_sessions = getattr(app.state, "rag_sessions", {})
        rag = rag_sessions.get(session_id)
 
        if rag is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}. Index documents first."
            )
 
        scores = ContractEvalSuite.run_eval_with_rag(rag)
 
        log.info("RAGAS evaluation complete", scores=scores)
        return {
            "session_id": session_id,
            "scores": scores,
            "interpretation": {
                "faithfulness": "How grounded the answers are in retrieved context",
                "answer_relevancy": "How well answers address the questions",
                "context_precision": "How relevant the retrieved chunks are",
                "context_recall": "How much relevant context was retrieved",
            }
        }
 
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    
# eval scores endpoint
@app.get("/eval/latest")
def get_latest_eval() -> Any:
    """Get the most recent evaluation scores — shown in README."""
    try:
        from src.eval.ragas_evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator()
        results = evaluator.load_latest_results()
        if results is None:
            return {"message": "No evaluation results yet. Run /eval/score first."}
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# command for executing the FastAPI app
# uvicorn api.main:app --port 8080 --reload    
# uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload