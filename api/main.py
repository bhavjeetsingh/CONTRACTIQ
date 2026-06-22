import os
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
from src.cache.redis_cache import cache
from src.document_chat.sse_streaming import router as stream_router, stream_rag_response
from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from constants import SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler
from logger import GLOBAL_LOGGER as log
from src.auth.jwt_handler import (
    UserCreate, Token, register_user,
    login_user, get_current_user, TokenData
)
import uuid
from pydantic import BaseModel

# Razorpay Config
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
razorpay_client = None
if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    try:
        import razorpay
        razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        log.info("Razorpay billing client initialized successfully")
    except Exception as e:
        log.error("Failed to initialize Razorpay client", error=str(e))
else:
    log.warning("Razorpay keys not set. Running billing in MOCK mode.")

class OrderRequest(BaseModel):
    tier: str

class PaymentVerifyRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

from src.database.supabase_client import is_supabase_active
from src.database.supabase_db import (
    check_user_limits, log_usage, save_rag_session,
    get_rag_session, save_extraction_results
)
from langchain_core.messages import HumanMessage, AIMessage

# Phase 1: LangGraph pipeline + Key-term extraction + Export
try:
    from src.agent.contract_graph import run_contract_pipeline
    LANGGRAPH_PIPELINE_AVAILABLE = True
except ImportError:
    LANGGRAPH_PIPELINE_AVAILABLE = False

try:
    from src.document_analyzer.key_term_extractor import KeyTermExtractor
    KEY_TERM_EXTRACTOR_AVAILABLE = True
except ImportError:
    KEY_TERM_EXTRACTOR_AVAILABLE = False

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

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
allow_all = "*" in ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else ALLOWED_ORIGINS,
    allow_credentials=not allow_all,
    allow_methods=["GET", "POST", "DELETE"],
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
        
        # Validate file size
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB"
            )
        await file.seek(0)
        
        dh = DocHandler()
        saved_path = dh.save_document(FastAPIFileAdapter(file))  # Updated to use save_document
        
        # Check monthly limits
        pages = dh.get_page_count(saved_path)
        if current_user.id and not check_user_limits(current_user.id):
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise HTTPException(status_code=402, detail="Monthly page limits exceeded. Please upgrade your subscription.")
            
        text = dh.read_document(saved_path)  # Updated to use read_document
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable")
        
        from src.document_analyzer.data_analysis import DocumentAnalyzer
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        
        # Log usage
        if current_user.id:
            log_usage(current_user.id, action="analyze", pages_processed=pages)
            
        log.info("Document analysis complete.", file_type=file_ext, content_length=len(text))
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis", filename=file.filename)
        raise HTTPException(status_code=500, detail="Analysis failed. Check server logs for details.")

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
        
        # Check monthly limits
        dh_temp = DocHandler(session_id=dc.session_id)
        ref_pages = dh_temp.get_page_count(ref_path)
        act_pages = dh_temp.get_page_count(act_path)
        total_pages = ref_pages + act_pages
        if current_user.id and not check_user_limits(current_user.id):
            if os.path.exists(ref_path): os.remove(ref_path)
            if os.path.exists(act_path): os.remove(act_path)
            raise HTTPException(status_code=402, detail="Monthly page limits exceeded. Please upgrade your subscription.")
            
        combined_text = dc.combine_documents()
        
        if not combined_text or not combined_text.strip():
            raise HTTPException(status_code=400, detail="One or both documents appear to be empty or unreadable")
        
        from src.document_compare.document_comparator import DocumentComparatorLLM
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        
        # Log usage
        if current_user.id:
            log_usage(current_user.id, action="compare", pages_processed=total_pages)
            
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
        raise HTTPException(status_code=500, detail="Comparison failed. Check server logs for details.")

# ---------- CHAT: INDEX (Hybrid RAG) ----------
@limiter.limit("5/minute")
@app.post("/chat/index")
async def chat_build_index(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1500),
    chunk_overlap: int = Form(300),
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
        
        # Check monthly limits
        dh_temp = DocHandler(session_id=ci.session_id)
        total_pages = 0
        for p in paths:
            total_pages += dh_temp.get_page_count(p)
        if current_user.id and not check_user_limits(current_user.id):
            for p in paths:
                if os.path.exists(p): os.remove(p)
            raise HTTPException(status_code=402, detail="Monthly page limits exceeded. Please upgrade your subscription.")
            
        docs = ci.prepare_documents(paths)
        
        # If not using ParentDocumentRetriever, we must chunk the docs here
        if not use_parent_retriever:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " "]
            )
            docs = splitter.split_documents(docs)
            log.info(f"Chunked documents into {len(docs)} chunks")
 
        faiss_dir = os.path.join(FAISS_BASE, ci.session_id) if use_session_dirs else FAISS_BASE
 
        from src.document_chat.hybrid_retrieval import ContractRAG
        rag = ContractRAG(
            session_id=ci.session_id,
            use_parent_retriever=use_parent_retriever,
            k=k,
        )
        rag.build(docs, faiss_dir=faiss_dir)
 
        # Store in app state (in-memory)
        app.state.rag_sessions = getattr(app.state, "rag_sessions", {})
        app.state.rag_sessions[ci.session_id] = rag
 
        session_meta = {
            "faiss_dir": faiss_dir,
            "k": k,
            "use_parent_retriever": use_parent_retriever,
            "files": [f.filename for f in files],
            "user": current_user.email,
        }
        # Store session metadata in Redis (survives restarts)
        cache.store_rag_session(ci.session_id, session_meta)
        
        # Store in Supabase DB
        if current_user.id:
            save_rag_session(ci.session_id, current_user.id, session_meta)
            log_usage(current_user.id, action="chat_index", pages_processed=total_pages)
 
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
        raise HTTPException(status_code=500, detail="Indexing failed. Check server logs for details.")
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

        # --- Load conversation history from Redis ---
        raw_history = cache.get_history(session_id, max_turns=20) if session_id else []
        chat_history = []
        for turn in raw_history:
            if turn["role"] == "user":
                chat_history.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                chat_history.append(AIMessage(content=turn["content"]))
 
        # --- Check cache first ---
        cached = cache.get(session_id, question, "hybrid")
        if cached:
            log.info("Returning cached response", session_id=session_id)
            cached["from_cache"] = True
            cached["messages"] = raw_history + [{"role": "user", "content": question}, {"role": "assistant", "content": cached.get("answer", "")}]
            return cached
 
        # --- Get RAG from app state ---
        rag_sessions = getattr(app.state, "rag_sessions", {})
        rag = rag_sessions.get(session_id)
 
        # --- Rebuild from Redis session metadata if app restarted ---
        if rag is None:
            session_meta = cache.get_rag_session(session_id)
            if not session_meta:
                session_meta = get_rag_session(session_id)
                if session_meta:
                    cache.store_rag_session(session_id, session_meta)
 
            if session_meta:
                log.info(f"Rebuilding RAG from Redis session metadata: {session_id}")
                from src.document_chat.hybrid_retrieval import ContractRAG
                rag = ContractRAG(
                    session_id=session_id,
                    use_parent_retriever=session_meta.get("use_parent_retriever", False),
                    k=session_meta.get("k", k),
                )
                faiss_dir = session_meta.get("faiss_dir")
                if faiss_dir and os.path.isdir(faiss_dir):
                    from langchain_community.vectorstores import FAISS
                    from utils.model_loader import ModelLoader
                    loader = ModelLoader()
                    embeddings = loader.load_embeddings()
                    vs = FAISS.load_local(
                        faiss_dir,
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    from src.document_chat.retrieval import ConversationalRAG
                    rag_fallback = ConversationalRAG(session_id=session_id)
                    rag_fallback.load_retriever_from_faiss(
                        faiss_dir, k=k, index_name=FAISS_INDEX_NAME
                    )
                    response = rag_fallback.invoke(question, chat_history=chat_history)
                    result = {
                        "answer": response,
                        "session_id": session_id,
                        "retriever_type": "basic-rebuild",
                        "from_cache": False,
                    }
                    cache.set(result, session_id, question, "hybrid")
                    cache.append_turn(session_id, "user", question)
                    cache.append_turn(session_id, "assistant", response)
                    result["messages"] = cache.get_history(session_id)
                    return result
 
            index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
            if not os.path.isdir(index_dir):
                raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
 
        # --- Invoke hybrid RAG with conversation history ---
        result = rag.invoke(question, chat_history=chat_history)
        result["from_cache"] = False
        
        if current_user.id:
            log_usage(current_user.id, action="chat_query", pages_processed=0)

        # --- Save turns to Redis ---
        answer_text = result.get("answer", "")
        cache.append_turn(session_id, "user", question)
        cache.append_turn(session_id, "assistant", answer_text)

        # --- Store response in cache ---
        cache.set(result, session_id, question, "hybrid")

        # --- Return full message history ---
        result["messages"] = cache.get_history(session_id)
 
        log.info("Hybrid RAG query handled successfully")
        return result
 
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail="Query failed. Check server logs for details.")

# ---------- CHAT: CLEAR HISTORY ----------
@app.post("/chat/new")
async def chat_new(
    request: Request,
    session_id: str = Form(...),
    current_user: TokenData = Depends(get_current_user)
) -> Any:
    cache.clear_history(session_id)
    return {"status": "ok", "session_id": session_id}

    
# ---------- UTILITY ENDPOINTS ----------

@app.get("/sessions")
async def list_sessions(current_user: TokenData = Depends(get_current_user)) -> Dict[str, Any]:
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
        raise HTTPException(status_code=500, detail="Failed to list sessions.")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user: TokenData = Depends(get_current_user)) -> Dict[str, str]:
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
        raise HTTPException(status_code=500, detail="Failed to delete session.")

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
# ---------- EXTRACT KEY TERMS (LangGraph Pipeline) ----------
@limiter.limit("10/minute")
@app.post("/extract")
async def extract_key_terms(
    request: Request,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
) -> Any:
    """
    Extract structured key terms from a contract using the LangGraph
    agentic pipeline with self-correcting extraction.
    
    Returns: parties, obligations, payment terms, risk flags,
    confidence scores, and more.
    """
    try:
        log.info(f"Key-term extraction requested: {file.filename}")
        
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")
        await file.seek(0)
        
        dh = DocHandler()
        saved_path = dh.save_document(FastAPIFileAdapter(file))
        
        # Check monthly limits
        pages = dh.get_page_count(saved_path)
        if current_user.id and not check_user_limits(current_user.id):
            if os.path.exists(saved_path): os.remove(saved_path)
            raise HTTPException(status_code=402, detail="Monthly page limits exceeded. Please upgrade your subscription.")
            
        if LANGGRAPH_PIPELINE_AVAILABLE:
            result = run_contract_pipeline(saved_path, request_type="extract")
            log.info("LangGraph extraction complete", method="langgraph")
        elif KEY_TERM_EXTRACTOR_AVAILABLE:
            text = dh.read_document(saved_path)
            if not text or not text.strip():
                raise HTTPException(status_code=400, detail="Document appears empty")
            extractor = KeyTermExtractor()
            result = extractor.extract(text)
            result["method"] = "direct"
            log.info("Direct extraction complete", method="direct")
        else:
            raise HTTPException(status_code=501, detail="Extraction pipeline not available")
            
        # Log usage & save extractions
        if current_user.id:
            log_usage(current_user.id, action="extract", pages_processed=pages)
            save_extraction_results(
                session_id=dh.session_id,
                user_id=current_user.id,
                extracted_terms=result.get("key_terms", {}) or result.get("extracted_terms", {}),
                confidence_scores=result.get("confidence_scores", {})
            )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Key-term extraction failed", filename=file.filename)
        raise HTTPException(status_code=500, detail="Extraction failed. Check server logs.")


# ---------- BILLING (Razorpay) ----------
@app.post("/billing/order")
def create_billing_order(
    order_req: OrderRequest,
    current_user: TokenData = Depends(get_current_user),
) -> Any:
    """
    Generate a Razorpay Order ID for upgrading to Premium subscription.
    """
    try:
        amount_paise = 49900  # ₹499 in paise
        
        if razorpay_client is not None:
            order_data = {
                "amount": amount_paise,
                "currency": "INR",
                "payment_capture": 1
            }
            order = razorpay_client.order.create(data=order_data)
            # Inject key id so frontend knows which key to open checkout with
            order["razorpay_key_id"] = RAZORPAY_KEY_ID
            log.info("Created live Razorpay order", order_id=order["id"], user=current_user.email)
            return order
        else:
            # Mock mode order response
            mock_order = {
                "id": f"order_mock_{uuid.uuid4().hex[:12]}",
                "amount": amount_paise,
                "currency": "INR",
                "razorpay_key_id": "rzp_test_mockkey123"
            }
            log.info("Created mock Razorpay order", order_id=mock_order["id"], user=current_user.email)
            return mock_order
            
    except Exception as e:
        log.exception("Failed to create billing order")
        raise HTTPException(status_code=500, detail="Failed to create checkout order.")


@app.post("/billing/verify")
def verify_billing_payment(
    verify_req: PaymentVerifyRequest,
    current_user: TokenData = Depends(get_current_user),
) -> Any:
    """
    Verify the signature returned by Razorpay checkouts, and upgrade the user profile tier.
    """
    try:
        signature_valid = False
        
        # Verify payment signature if running live
        if razorpay_client is not None:
            try:
                razorpay_client.utility.verify_payment_signature({
                    'razorpay_order_id': verify_req.razorpay_order_id,
                    'razorpay_payment_id': verify_req.razorpay_payment_id,
                    'razorpay_signature': verify_req.razorpay_signature
                })
                signature_valid = True
                log.info("Razorpay signature verified successfully", order_id=verify_req.razorpay_order_id)
            except Exception as e:
                log.warning("Razorpay signature verification failed", error=str(e))
                raise HTTPException(status_code=400, detail="Invalid payment signature.")
        else:
            # Mock verify: accept mock payment ids
            if verify_req.razorpay_order_id.startswith("order_mock_"):
                signature_valid = True
                log.info("Mock Razorpay order verified successfully", order_id=verify_req.razorpay_order_id)
            else:
                log.warning("Mock Razorpay validation received non-mock order id", order_id=verify_req.razorpay_order_id)
                raise HTTPException(status_code=400, detail="Invalid mock order format.")
                
        if signature_valid:
            # Upgrade database profiles tier to premium
            if is_supabase_active():
                supabase_client.table("profiles").update({"subscription_tier": "premium"}).eq("id", current_user.id).execute()
                log.info("Upgraded user profile in Supabase to premium", user_id=current_user.id)
            else:
                # Mock profiles file edit
                from src.database.supabase_db import MOCK_PROFILES_FILE, _load_mock_file, _save_mock_file
                db = _load_mock_file(MOCK_PROFILES_FILE)
                if current_user.id in db:
                    db[current_user.id]["subscription_tier"] = "premium"
                    _save_mock_file(MOCK_PROFILES_FILE, db)
                    log.info("Upgraded user profile in Mock DB to premium", user_id=current_user.id)
                    
            return {"status": "verified", "tier": "premium"}
            
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Payment verification exception")
        raise HTTPException(status_code=500, detail="Internal validation failure.")


# ---------- EXPORT ----------
@app.get("/export/{session_id}/{format}")
async def export_results(
    session_id: str,
    format: str,
    current_user: TokenData = Depends(get_current_user),
) -> Any:
    """
    Export analysis/extraction results in JSON, CSV, or PDF format.
    
    Args:
        session_id: Session ID from a previous analysis
        format: Export format — "json", "csv", or "pdf"
    """
    try:
        if format not in ("json", "csv", "pdf"):
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Use json, csv, or pdf.")
        
        # Get cached results from Redis
        session_meta = cache.get_rag_session(session_id)
        if not session_meta:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if format == "json":
            from src.export.json_exporter import export_json
            json_str = export_json(session_meta)
            return JSONResponse(content={"export": json_str, "format": "json"})
        
        elif format == "csv":
            from src.export.csv_exporter import export_csv
            csv_str = export_csv(session_meta)
            from fastapi.responses import Response
            return Response(
                content=csv_str,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=contractiq_{session_id}.csv"},
            )
        
        elif format == "pdf":
            from src.export.pdf_exporter import export_pdf
            pdf_bytes = export_pdf(session_meta)
            from fastapi.responses import Response
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=contractiq_{session_id}.pdf"},
            )
    
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Export failed", session_id=session_id, format=format)
        raise HTTPException(status_code=500, detail="Export failed. Check server logs.")


# ---------- ANALYZE WITH LANGGRAPH ----------
@limiter.limit("10/minute")
@app.post("/analyze/v2")
async def analyze_document_v2(
    request: Request,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
) -> Any:
    """
    V2 analysis endpoint using LangGraph agentic pipeline.
    
    Returns both metadata analysis AND key-term extraction
    with confidence scores and self-correcting validation.
    Falls back to v1 linear pipeline if LangGraph unavailable.
    """
    try:
        log.info(f"V2 analysis requested: {file.filename}")
        
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")
        await file.seek(0)
        
        dh = DocHandler()
        saved_path = dh.save_document(FastAPIFileAdapter(file))
        
        # Check monthly limits
        pages = dh.get_page_count(saved_path)
        if current_user.id and not check_user_limits(current_user.id):
            if os.path.exists(saved_path): os.remove(saved_path)
            raise HTTPException(status_code=402, detail="Monthly page limits exceeded. Please upgrade your subscription.")
            
        if LANGGRAPH_PIPELINE_AVAILABLE:
            result = run_contract_pipeline(saved_path, request_type="analyze")
            log.info("V2 LangGraph analysis complete")
        else:
            # Fallback to v1
            text = dh.read_document(saved_path)
            if not text or not text.strip():
                raise HTTPException(status_code=400, detail="Document appears empty")
            from src.document_analyzer.data_analysis import DocumentAnalyzer
            analyzer = DocumentAnalyzer()
            result = analyzer.analyze_document(text)
            result["method"] = "legacy_v1"
            log.info("V2 fallback to v1 analysis")
            
        # Log usage & save extractions
        if current_user.id:
            log_usage(current_user.id, action="analyze_v2", pages_processed=pages)
            key_terms = result.get("key_terms", {}) or result.get("extracted_terms", {})
            conf_scores = result.get("confidence_scores", {})
            if key_terms:
                save_extraction_results(
                    session_id=dh.session_id,
                    user_id=current_user.id,
                    extracted_terms=key_terms,
                    confidence_scores=conf_scores
                )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        log.exception("V2 analysis failed", filename=file.filename)
        raise HTTPException(status_code=500, detail="Analysis failed. Check server logs.")


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
 
        from src.eval.ragas_evaluator import ContractEvalSuite
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
        raise HTTPException(status_code=500, detail="Evaluation failed. Check server logs.")
    
# eval scores endpoint
@app.get("/eval/latest")
def get_latest_eval() -> Any:
    """Get the most recent evaluation scores — shown in README."""
    try:
        from src.eval.ragas_evaluator import ContractEvalSuite
        import json
        from pathlib import Path
        
        results_dir = Path("eval_results")
        if results_dir.exists():
            files = list(results_dir.glob("eval_output_*.json"))
            if files:
                latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
                with open(latest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
                    
        return {"message": "No evaluation results yet. Run python run_eval.py first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# command for executing the FastAPI app
# uvicorn api.main:app --port 8080 --reload    
# uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload