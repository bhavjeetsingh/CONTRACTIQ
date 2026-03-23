"""
SSE Streaming for ContractIQ
=====================================
Streams LLM tokens word by word to the client.
Instead of waiting 3-4 seconds for full response,
user sees tokens appear in real time like ChatGPT.

How it works:
1. Client sends question via POST /chat/stream
2. Server opens a Server-Sent Events connection
3. LLM tokens stream back one by one as they generate
4. Client receives and displays each token immediately
5. Connection closes when generation is complete

Why this matters:
- Perceived latency drops from 3-4s to <200ms (first token)
- Standard expectation in any modern AI product
- Directly demonstrates production AI backend knowledge
"""

import os
import json
import asyncio
from typing import AsyncGenerator, Optional, List
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from src.auth.jwt_handler import get_current_user, TokenData
from src.cache.redis_cache import cache
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

router = APIRouter()

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")


class StreamingCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler that captures LLM tokens as they stream.
    Puts each token into an asyncio queue for the SSE endpoint to consume.
    """

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called by LangChain for each new token from the LLM."""
        await self.queue.put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes generating."""
        self.done = True
        await self.queue.put(None)  # Sentinel value to signal end

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called if LLM errors during generation."""
        log.error("LLM streaming error", error=str(error))
        await self.queue.put(None)

    async def token_generator(self) -> AsyncGenerator[str, None]:
        """
        Async generator that yields tokens as SSE events.
        Yields None when generation is complete.
        """
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token


async def stream_rag_response(
    question: str,
    session_id: str,
    app_state,
    k: int = 5,
) -> AsyncGenerator[str, None]:
    """
    Core streaming generator for RAG responses.

    Yields SSE-formatted events:
    - data: {"token": "word"} for each token
    - data: {"done": true} when complete
    - data: {"error": "message"} on failure
    """
    try:
        # Check cache first — stream cached response instantly
        cached = cache.get(session_id, question, "stream")
        if cached and cached.get("answer"):
            log.info("Streaming cached response", session_id=session_id)
            answer = cached["answer"]
            # Stream cached tokens with small delay for natural feel
            words = answer.split(" ")
            for word in words:
                token_event = json.dumps({"token": word + " "})
                yield f"data: {token_event}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'done': True, 'from_cache': True})}\n\n"
            return

        # Get RAG instance from app state
        rag_sessions = getattr(app_state, "rag_sessions", {})
        rag = rag_sessions.get(session_id)

        if rag is None:
            # Try rebuilding from saved FAISS index
            index_dir = os.path.join(FAISS_BASE, session_id)
            if not os.path.isdir(index_dir):
                yield f"data: {json.dumps({'error': f'Session not found: {session_id}'})}\n\n"
                return

            # Fall back to basic ConversationalRAG for rebuild
            from src.document_chat.retrieval import ConversationalRAG
            rag_fallback = ConversationalRAG(session_id=session_id)
            rag_fallback.load_retriever_from_faiss(
                index_dir, k=k, index_name=FAISS_INDEX_NAME
            )

            # Stream the fallback response
            callback = StreamingCallbackHandler()
            try:
                # Run in background and stream tokens
                task = asyncio.create_task(
                    rag_fallback.ainvoke(question, chat_history=[], callbacks=[callback])
                )
                full_response = ""
                async for token in callback.token_generator():
                    full_response += token
                    token_event = json.dumps({"token": token})
                    yield f"data: {token_event}\n\n"

                await task
                cache.set(
                    {"answer": full_response},
                    session_id, question, "stream"
                )
                yield f"data: {json.dumps({'done': True, 'from_cache': False})}\n\n"
                return

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

        # Stream from hybrid RAG
        callback = StreamingCallbackHandler()
        full_response = ""

        try:
            # Inject streaming callback into LLM
            rag.llm.callbacks = [callback]

            # Run RAG invoke in background task
            task = asyncio.create_task(
                asyncio.to_thread(rag.invoke, question, [])
            )

            # Stream tokens as they arrive
            async for token in callback.token_generator():
                full_response += token
                token_event = json.dumps({"token": token})
                yield f"data: {token_event}\n\n"

            result = await task

            # Cache the complete response
            cache.set(
                {"answer": full_response},
                session_id, question, "stream"
            )

            yield f"data: {json.dumps({'done': True, 'from_cache': False})}\n\n"

        except Exception as e:
            log.error("Streaming generation failed", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    except Exception as e:
        log.error("SSE stream failed", error=str(e))
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


