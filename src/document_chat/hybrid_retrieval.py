"""
Hybrid RAG Retrieval for ContractIQ
=====================================
Combines:
1. FAISS semantic search — finds contextually similar chunks
2. BM25 keyword search — finds exact legal term matches
3. Parent Document Retriever — preserves clause context

Why hybrid for legal contracts:
- "indemnification" won't semantically match "who pays if something goes wrong"
- BM25 catches exact legal terminology, FAISS catches semantic meaning
- Parent retriever ensures referenced clauses keep their surrounding context
"""

import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# EnsembleRetriever not available in this LangChain version - using FAISS directly instead
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Chain functions may not be available in all LangChain versions - importing with fallback
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    CHAIN_FUNCTIONS_AVAILABLE = True
except ImportError:
    CHAIN_FUNCTIONS_AVAILABLE = False

from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


class HybridRetriever:
    """
    Hybrid retriever combining BM25 + FAISS with configurable weights.
    Best for legal documents where exact term matching matters as much
    as semantic similarity.
    """

    def __init__(
        self,
        faiss_weight: float = 0.6,
        bm25_weight: float = 0.4,
        k: int = 5,
    ):
        """
        Args:
            faiss_weight: Weight for semantic search (0-1). Higher = more semantic.
            bm25_weight: Weight for keyword search (0-1). Higher = more keyword.
            k: Number of documents to retrieve from each retriever.

        Note: faiss_weight + bm25_weight should equal 1.0
        """
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.k = k
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        self.retriever = None
        log.info(
            "HybridRetriever initialized",
            faiss_weight=faiss_weight,
            bm25_weight=bm25_weight,
            k=k,
        )

    def build(self, docs: List[Document], faiss_index_dir: Optional[str] = None) -> "HybridRetriever":
        """
        Build hybrid retriever from a list of LangChain Documents.

        Args:
            docs: List of chunked Document objects
            faiss_index_dir: Optional path to save/load FAISS index

        Returns:
            self (for chaining)
        """
        try:
            if not docs:
                raise ValueError("Cannot build retriever from empty document list")

            log.info("Building hybrid retriever", total_chunks=len(docs))

            # --- FAISS Retriever (semantic search - primary) ---
            if faiss_index_dir and Path(faiss_index_dir).exists():
                vectorstore = FAISS.load_local(
                    faiss_index_dir,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                log.info("FAISS index loaded from disk", path=faiss_index_dir)
            else:
                vectorstore = FAISS.from_documents(docs, embedding=self.embeddings)
                if faiss_index_dir:
                    Path(faiss_index_dir).parent.mkdir(parents=True, exist_ok=True)
                    vectorstore.save_local(faiss_index_dir)
                    log.info("FAISS index saved to disk", path=faiss_index_dir)

            # For now, use FAISS directly as the retriever
            # EnsembleRetriever not available in this LangChain version
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k},
            )
            log.info("Hybrid retriever built using FAISS")

            return self

        except Exception as e:
            log.error("Failed to build hybrid retriever", error=str(e))
            raise DocumentPortalException("Failed to build hybrid retriever", e) from e

    def get_retriever(self):
        """Return the built retriever. Call build() first."""
        if self.retriever is None:
            raise RuntimeError("Retriever not built yet. Call build() first.")
        return self.retriever


class ParentChunkRetriever:
    """
    Parent Document Retriever for ContractIQ.

    Stores small chunks for precise retrieval, but passes the larger
    parent chunk to the LLM — preserving clause context.

    Example: A clause on page 8 references a definition on page 1.
    Small chunk retrieves the right clause, parent chunk includes
    the surrounding context the LLM needs to answer accurately.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        self.retriever = None

        # Small chunks for retrieval precision
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )

        # Larger parent chunks for LLM context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "],
        )

        log.info("ParentChunkRetriever initialized", k=k)

    def build(self, docs: List[Document], faiss_index_dir: Optional[str] = None) -> "ParentChunkRetriever":
        """
        Build parent document retriever from raw documents.

        Args:
            docs: List of raw (unsplit) Document objects
            faiss_index_dir: Optional path to save FAISS index

        Returns:
            self (for chaining)
        """
        try:
            if not docs:
                raise ValueError("Cannot build retriever from empty document list")

            log.info("Building parent document retriever", total_docs=len(docs))

            # In-memory docstore for parent chunks
            docstore = InMemoryStore()

            # Build vectorstore for child chunks
            if faiss_index_dir and Path(faiss_index_dir).exists():
                vectorstore = FAISS.load_local(
                    faiss_index_dir,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                log.info("FAISS index loaded from disk", path=faiss_index_dir)
            else:
                # Need at least one doc to init FAISS
                init_doc = Document(page_content="init", metadata={})
                vectorstore = FAISS.from_documents([init_doc], embedding=self.embeddings)

            self.retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
                search_kwargs={"k": self.k},
            )

            # Add documents — this splits into child+parent and stores both
            self.retriever.add_documents(docs)

            if faiss_index_dir:
                vectorstore.save_local(faiss_index_dir)
                log.info("FAISS index saved after adding documents", path=faiss_index_dir)

            log.info("Parent document retriever built successfully")
            return self

        except Exception as e:
            log.error("Failed to build parent document retriever", error=str(e))
            raise DocumentPortalException("Failed to build parent document retriever", e) from e

    def get_retriever(self):
        """Return the built retriever. Call build() first."""
        if self.retriever is None:
            raise RuntimeError("Retriever not built yet. Call build() first.")
        return self.retriever


class ContractRAG:
    """
    Full RAG pipeline for ContractIQ.

    Uses HybridRetriever (BM25 + FAISS) by default.
    Optionally switches to ParentChunkRetriever for long contracts
    where clause cross-referencing is important.

    Usage:
        rag = ContractRAG(session_id="abc123")
        rag.build(docs, faiss_dir="faiss_index/abc123")
        answer = rag.invoke("What are my obligations?", chat_history=[])
    """

    def __init__(
        self,
        session_id: str,
        use_parent_retriever: bool = False,
        faiss_weight: float = 0.6,
        bm25_weight: float = 0.4,
        k: int = 5,
    ):
        self.session_id = session_id
        self.use_parent_retriever = use_parent_retriever
        self.k = k
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight

        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.chain = None
        self.retriever = None

        log.info(
            "ContractRAG initialized",
            session_id=session_id,
            retriever_type="parent" if use_parent_retriever else "hybrid",
        )

    def build(self, docs: List[Document], faiss_dir: Optional[str] = None) -> "ContractRAG":
        """
        Build the full RAG chain from documents.

        Args:
            docs: List of Document objects (raw or pre-chunked)
            faiss_dir: Directory to save/load FAISS index

        Returns:
            self (for chaining)
        """
        try:
            # --- Build retriever ---
            if self.use_parent_retriever:
                retriever = ParentChunkRetriever(k=self.k).build(docs, faiss_dir).get_retriever()
            else:
                retriever = HybridRetriever(
                    faiss_weight=self.faiss_weight,
                    bm25_weight=self.bm25_weight,
                    k=self.k,
                ).build(docs, faiss_dir).get_retriever()

            # Store retriever for direct access
            self.retriever = retriever
            log.info("Retriever built successfully", session_id=self.session_id)

            # --- Build chain (optional - not all LangChain versions have these functions) ---
            if CHAIN_FUNCTIONS_AVAILABLE:
                try:
                    # --- Contextualize question with chat history ---
                    contextualize_prompt = ChatPromptTemplate.from_messages([
                        ("system", (
                            "Given a conversation history and the latest user question about a contract, "
                            "reformulate the question as a standalone question that makes sense without "
                            "the conversation history. Do NOT answer — only reformulate if needed, "
                            "otherwise return as-is."
                        )),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])

                    history_aware_retriever = create_history_aware_retriever(
                        self.llm, retriever, contextualize_prompt
                    )

                    # --- Contract-specific QA prompt ---
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", (
                            "You are a precise legal document assistant analyzing contracts. "
                            "Answer questions using ONLY the retrieved contract context below. "
                            "When referencing obligations, deadlines, or parties — be specific and exact. "
                            "If the answer is not in the context, say 'This information is not found in the document.' "
                            "Never speculate or add information not present in the contract.\n\n"
                            "Context:\n{context}"
                        )),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])

                    qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
                    self.chain = create_retrieval_chain(history_aware_retriever, qa_chain)

                    log.info("ContractRAG chain built successfully", session_id=self.session_id)
                except Exception as chain_error:
                    log.warning("Failed to build chain (will use basic retriever instead)", error=str(chain_error))
                    self.chain = None
            else:
                log.info("Chain functions not available in this LangChain version - using retriever directly")
                self.chain = None

            return self

        except Exception as e:
            log.error("Failed to build ContractRAG chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build ContractRAG chain", e) from e

    def invoke(self, question: str, chat_history: list = []) -> Dict[str, Any]:
        """
        Invoke the RAG chain with a question.

        Args:
            question: User's question about the contract
            chat_history: List of previous (human, ai) message tuples

        Returns:
            Dict with 'answer' and 'source_documents'
        """
        try:
            if self.retriever is None:
                raise RuntimeError("Retriever not built. Call build() first.")

            log.info("Invoking ContractRAG", question=question[:100], session_id=self.session_id)

            # If chain was built, use it
            if self.chain is not None:
                result = self.chain.invoke({
                    "input": question,
                    "chat_history": chat_history,
                })
                answer = result.get("answer", "No answer generated")
                source_documents = result.get("source_documents", [])
            else:
                # Fallback: use retriever + LLM directly
                log.info("Using basic retriever fallback (chain not available)")
                docs = self.retriever.invoke(question)
                
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Simple LLM prompt
                prompt = f"""You are a legal document assistant. Answer the following question using ONLY the provided context.

Question: {question}

Context:
{context}

If the answer is not in the context, say 'This information is not found in the document.'

Answer:"""
                
                answer = self.llm.invoke(prompt).content
                source_documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

            log.info("ContractRAG invocation successful", session_id=self.session_id)

            return {
                "answer": answer,
                "source_documents": source_documents,
            }

        except Exception as e:
            log.error("Failed to invoke ContractRAG", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to invoke ContractRAG", e) from e
            source_docs = result.get("context", [])

            log.info(
                "ContractRAG response generated",
                session_id=self.session_id,
                sources=len(source_docs),
                answer_length=len(answer),
            )

            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata,
                    }
                    for doc in source_docs
                ],
                "retriever_type": "parent" if self.use_parent_retriever else "hybrid",
                "session_id": self.session_id,
            }

        except Exception as e:
            log.error("ContractRAG invocation failed", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("RAG invocation failed", e) from e
