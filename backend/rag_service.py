import os
import re
import json
import uuid
import shutil
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "backend" / ".env")

# Configuration constants
RAG_STORAGE_DIR = Path(os.getenv("RAG_STORAGE_DIR", str(BASE_DIR / "data" / "rag")))
RAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
MAX_FILE_SIZE = int(os.getenv("RAG_MAX_FILE_SIZE", str(25 * 1024 * 1024)))  # 25 MB default limit


def sanitize_id(identifier: str) -> str:
    """Sanitize thread or document IDs to prevent path traversal vulnerabilities."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", str(identifier))
    if not cleaned:
        raise ValueError("Invalid identifier: contains no valid alphanumeric characters.")
    return cleaned


def get_embeddings(api_key: Optional[str] = None):
    """
    Get configured embedding model.
    Uses OpenAIEmbeddings when a valid live key is provided (either directly or via thread context),
    or a deterministic FakeEmbeddings fallback for offline development and automated testing doubles.
    """
    if os.getenv("RAG_USE_TEST_EMBEDDINGS", "false").lower() in ("true", "1", "yes"):
        from langchain_core.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=1536)

    try:
        from backend.tools import get_active_openai_key
        ctx_key = get_active_openai_key()
    except Exception:
        ctx_key = None

    openai_key = api_key or ctx_key or os.getenv("OPENAI_API_KEY")
    # Ignore known dummy/placeholder keys
    if openai_key and not openai_key.startswith("sk-ijklmnop") and openai_key != "Gareebon":
        try:
            from langchain_openai import OpenAIEmbeddings
            model_name = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
            return OpenAIEmbeddings(model=model_name, api_key=openai_key)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAIEmbeddings: {e}")

    # Fallback to test double embeddings when external API keys are unavailable
    from langchain_core.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=1536)


class RAGService:
    """
    Modular RAG Service providing:
    - PDF ingestion & text extraction
    - Recursive chunking & vector embedding
    - Persistent disk-backed FAISS vector storage per thread
    - Strict thread isolation and document metadata tracking
    - Excerpt retrieval with source citation metadata
    """

    def __init__(self, storage_dir: Path = RAG_STORAGE_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # In-memory index cache: {f"{thread_id}_{doc_id}": faiss_vectorstore}
        self._cache: Dict[str, FAISS] = {}

    def _get_thread_dir(self, thread_id: str) -> Path:
        safe_tid = sanitize_id(thread_id)
        tdir = self.storage_dir / safe_tid
        tdir.mkdir(parents=True, exist_ok=True)
        return tdir

    def _get_metadata_path(self, thread_id: str) -> Path:
        return self._get_thread_dir(thread_id) / "metadata.json"

    def _load_metadata(self, thread_id: str) -> Dict[str, Any]:
        mpath = self._get_metadata_path(thread_id)
        if mpath.exists():
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata for thread {thread_id}: {e}")
        return {}

    def _save_metadata(self, thread_id: str, data: Dict[str, Any]):
        mpath = self._get_metadata_path(thread_id)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def list_documents(self, thread_id: str) -> List[Dict[str, Any]]:
        """List all indexed documents for a given thread."""
        meta = self._load_metadata(thread_id)
        return sorted(list(meta.values()), key=lambda d: d.get("upload_time", ""), reverse=True)

    def ingest_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        thread_id: str,
        openai_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate, extract, chunk, embed, and persist a PDF document for a specific thread.
        """
        safe_tid = sanitize_id(thread_id)
        
        # 1. Validation
        if not file_bytes:
            raise ValueError("Uploaded file is empty (0 bytes).")
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE // (1024*1024)} MB.")
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF documents (.pdf) are supported.")

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        thread_dir = self._get_thread_dir(safe_tid)
        index_dir = thread_dir / doc_id

        # 2. Extract text using pypdf
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        documents: List[Document] = []
        try:
            reader = PdfReader(tmp_path)
            if len(reader.pages) == 0:
                raise ValueError("PDF document contains no pages.")

            total_extracted_text = ""
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                total_extracted_text += page_text
                if page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": filename,
                                "page": idx + 1,
                                "document_id": doc_id,
                                "thread_id": safe_tid,
                            },
                        )
                    )

            if not total_extracted_text.strip() or not documents:
                raise ValueError("Could not extract any readable text from this PDF (it may be scanned or empty).")

            # 3. Chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_documents(documents)
            if not chunks:
                raise ValueError("No text chunks generated from document.")

            # 4. Generate Embeddings & FAISS Index
            embeddings = get_embeddings(api_key=openai_api_key)
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # 5. Persist FAISS Index to disk
            index_dir.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(index_dir))

            # Cache in memory
            cache_key = f"{safe_tid}_{doc_id}"
            self._cache[cache_key] = vectorstore

            try:
                rel_index_path = str(index_dir.relative_to(BASE_DIR))
            except ValueError:
                rel_index_path = str(index_dir)

            doc_meta = {
                "document_id": doc_id,
                "thread_id": safe_tid,
                "filename": filename,
                "upload_time": datetime.now(timezone.utc).isoformat(),
                "chunks": len(chunks),
                "pages": len(reader.pages),
                "size_bytes": len(file_bytes),
                "index_path": rel_index_path,
            }

            all_meta = self._load_metadata(safe_tid)
            all_meta[doc_id] = doc_meta
            self._save_metadata(safe_tid, all_meta)

            logger.info(f"Indexed PDF '{filename}' ({doc_id}) with {len(chunks)} chunks for thread '{safe_tid}'.")
            return doc_meta

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def load_index_for_doc(self, thread_id: str, doc_id: str) -> Optional[FAISS]:
        """
        Load a specific document's FAISS index safely from disk with cache optimization.
        Security Note: allow_dangerous_deserialization is set to True strictly for application-managed
        internal indexes verified to reside in the authorized thread directory.
        """
        safe_tid = sanitize_id(thread_id)
        safe_doc_id = sanitize_id(doc_id)
        cache_key = f"{safe_tid}_{safe_doc_id}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        index_dir = self._get_thread_dir(safe_tid) / safe_doc_id
        if not index_dir.exists() or not (index_dir / "index.faiss").exists():
            return None

        try:
            embeddings = get_embeddings()
            vs = FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            self._cache[cache_key] = vs
            return vs
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_dir}: {e}")
            return None

    def query_documents(self, thread_id: str, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Query all indexed documents associated with a thread and return relevant excerpts with source citations.
        """
        safe_tid = sanitize_id(thread_id)
        docs_meta = self.list_documents(safe_tid)

        if not docs_meta:
            return {
                "error": "No documents uploaded for this chat thread. Please upload a PDF first to use document search.",
                "query": query,
                "excerpts": [],
            }

        all_matches = []
        for doc in docs_meta:
            doc_id = doc["document_id"]
            vs = self.load_index_for_doc(safe_tid, doc_id)
            if vs:
                try:
                    results = vs.similarity_search_with_score(query, k=k)
                    for chunk, score in results:
                        all_matches.append({
                            "content": chunk.page_content,
                            "source": chunk.metadata.get("source", doc.get("filename")),
                            "page": chunk.metadata.get("page", 1),
                            "document_id": doc_id,
                            "score": float(score),
                        })
                except Exception as e:
                    logger.warning(f"Error querying doc {doc_id}: {e}")

        if not all_matches:
            return {
                "message": "No relevant content found in the uploaded documents for this query.",
                "query": query,
                "excerpts": [],
            }

        # Sort by distance score ascending (lower is closer in FAISS L2)
        all_matches.sort(key=lambda m: m["score"])
        top_matches = all_matches[:k]

        return {
            "query": query,
            "thread_id": safe_tid,
            "excerpts": [
                {
                    "content": m["content"],
                    "source": m["source"],
                    "page": m["page"],
                }
                for m in top_matches
            ],
            "total_matches": len(top_matches),
        }

    def delete_document(self, thread_id: str, doc_id: str) -> bool:
        """Delete document index and metadata for a thread."""
        safe_tid = sanitize_id(thread_id)
        safe_doc_id = sanitize_id(doc_id)
        cache_key = f"{safe_tid}_{safe_doc_id}"

        # Evict from cache
        self._cache.pop(cache_key, None)

        thread_dir = self._get_thread_dir(safe_tid)
        index_dir = thread_dir / safe_doc_id

        if index_dir.exists():
            shutil.rmtree(index_dir, ignore_errors=True)

        all_meta = self._load_metadata(safe_tid)
        if safe_doc_id in all_meta:
            del all_meta[safe_doc_id]
            self._save_metadata(safe_tid, all_meta)
            logger.info(f"Deleted document {safe_doc_id} from thread {safe_tid}.")
            return True

        return False


# Singleton instance
rag_service = RAGService()
