import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.chatbot import (
    execute_chat,
    retrieve_all_threads,
    get_thread_messages,
    build_chatbot_graph,
)
from backend.mcp_manager import mcp_manager
from backend.rag_service import rag_service, sanitize_id, MAX_FILE_SIZE

logger = logging.getLogger("backend.main")
logging.basicConfig(level=logging.INFO)


# Lifespan context manager for MCP discovery and graceful resource cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application: Discovering MCP tools...")
    try:
        await mcp_manager.initialize()
        all_tools = mcp_manager.get_all_tools()
        build_chatbot_graph(all_tools)
        logger.info(f"Graph initialized with {len(all_tools)} tools ({len(mcp_manager.discovered_mcp_tools)} MCP).")
    except Exception as exc:
        logger.error(f"Error during MCP initialization: {exc}", exc_info=True)
    yield
    logger.info("Shutting down FastAPI application: Cleaning up MCP resources...")
    await mcp_manager.cleanup()


# Initialize FastAPI application
app = FastAPI(
    title="LangGraph Chatbot API with MCP & Persistent PDF RAG",
    description="FastAPI backend exposing LangGraph assistant with native tools, MCP tools, persistent FAISS RAG, and SQLite checkpoints",
    version="1.2.0",
    lifespan=lifespan,
)

# Configure CORS for local React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Schemas
# ============================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message text to process")
    thread_id: Optional[str] = Field(None, description="Existing thread ID or None for a new chat session")
    openai_api_key: Optional[str] = Field(None, description="Optional user-provided OpenAI API key")
    hf_token: Optional[str] = Field(None, description="Optional user-provided Hugging Face token")

    @field_validator("message")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message cannot be empty or whitespace only.")
        return trimmed


class ChatResponse(BaseModel):
    thread_id: str
    response: str


class ThreadsResponse(BaseModel):
    threads: List[str]


class MessageItem(BaseModel):
    role: str
    content: str


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: List[MessageItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    mcp_servers_online: int


class DocumentResponse(BaseModel):
    document_id: str
    thread_id: str
    filename: str
    upload_time: str
    chunks: int
    pages: int
    size_bytes: int


class DocumentsListResponse(BaseModel):
    thread_id: str
    documents: List[DocumentResponse]


# ============================================================================
# Core Chat & Health Endpoints
# ============================================================================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Return backend health status and count of online MCP servers."""
    statuses = mcp_manager.server_statuses
    online_count = sum(1 for s in statuses.values() if s.get("connected"))
    return HealthResponse(
        status="ok",
        version="1.2.0",
        mcp_servers_online=online_count,
    )


@app.get("/api/tools")
async def get_tools_endpoint():
    """Return all registered tools and MCP server connection statuses."""
    return mcp_manager.get_status_report()


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    x_openai_api_key: Optional[str] = Header(None, alias="X-OpenAI-API-Key"),
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token"),
):
    """Execute chat turn via LangGraph with support for user-provided API keys."""
    thread_id = request.thread_id.strip() if request.thread_id and request.thread_id.strip() else str(uuid.uuid4())
    effective_openai_key = (request.openai_api_key or x_openai_api_key or "").strip() or None
    effective_hf_token = (request.hf_token or x_hf_token or "").strip() or None

    try:
        assistant_reply = await asyncio.to_thread(
            execute_chat,
            request.message,
            thread_id,
            effective_openai_key,
            effective_hf_token,
        )
        return ChatResponse(thread_id=thread_id, response=assistant_reply)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat execution failed: {str(exc)}",
        )


@app.get("/api/threads", response_model=ThreadsResponse)
async def get_threads_endpoint():
    """Retrieve all available conversation thread IDs from SQLite checkpoints."""
    try:
        threads = await asyncio.to_thread(retrieve_all_threads)
        return ThreadsResponse(threads=threads)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conversation threads: {str(exc)}",
        )


@app.get("/api/threads/{thread_id}", response_model=ThreadHistoryResponse)
async def get_thread_history_endpoint(thread_id: str):
    """Retrieve the saved message history for a given conversation thread ID."""
    clean_id = thread_id.strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Thread ID cannot be empty.",
        )

    try:
        messages = await asyncio.to_thread(get_thread_messages, clean_id)
        return ThreadHistoryResponse(
            thread_id=clean_id,
            messages=[MessageItem(**msg) for msg in messages],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve thread history: {str(exc)}",
        )


# ============================================================================
# Persistent PDF RAG Endpoints
# ============================================================================
@app.post("/api/threads/{thread_id}/documents", response_model=DocumentResponse)
async def upload_document_endpoint(
    thread_id: str,
    file: UploadFile = File(...),
    x_openai_api_key: Optional[str] = Header(None, alias="X-OpenAI-API-Key"),
):
    """
    Upload and index a PDF document for a specific chat thread.
    Chunks text, computes vector embeddings, and persists the FAISS index to disk.
    """
    try:
        clean_tid = sanitize_id(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files (.pdf) are supported.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded PDF exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    try:
        # Offload CPU-heavy text extraction and embedding to threadpool
        doc_meta = await asyncio.to_thread(
            rag_service.ingest_pdf,
            file_bytes,
            file.filename,
            clean_tid,
            (x_openai_api_key or "").strip() or None,
        )

        return DocumentResponse(
            document_id=doc_meta["document_id"],
            thread_id=doc_meta["thread_id"],
            filename=doc_meta["filename"],
            upload_time=doc_meta["upload_time"],
            chunks=doc_meta["chunks"],
            pages=doc_meta["pages"],
            size_bytes=doc_meta["size_bytes"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.error(f"PDF ingestion failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index PDF: {str(exc)}",
        )


@app.get("/api/threads/{thread_id}/documents", response_model=DocumentsListResponse)
async def list_documents_endpoint(thread_id: str):
    """Retrieve all indexed documents associated with a specific chat thread."""
    try:
        clean_tid = sanitize_id(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        docs = await asyncio.to_thread(rag_service.list_documents, clean_tid)
        return DocumentsListResponse(
            thread_id=clean_tid,
            documents=[
                DocumentResponse(
                    document_id=d["document_id"],
                    thread_id=d["thread_id"],
                    filename=d["filename"],
                    upload_time=d["upload_time"],
                    chunks=d["chunks"],
                    pages=d["pages"],
                    size_bytes=d["size_bytes"],
                )
                for d in docs
            ],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch documents: {str(exc)}",
        )


@app.delete("/api/threads/{thread_id}/documents/{document_id}")
async def delete_document_endpoint(thread_id: str, document_id: str):
    """Delete an indexed document and its persistent FAISS index from disk."""
    try:
        clean_tid = sanitize_id(thread_id)
        clean_doc_id = sanitize_id(document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    deleted = await asyncio.to_thread(rag_service.delete_document, clean_tid, clean_doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{clean_doc_id}' not found for thread '{clean_tid}'.",
        )

    return {
        "status": "success",
        "deleted": True,
        "document_id": clean_doc_id,
        "thread_id": clean_tid,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
