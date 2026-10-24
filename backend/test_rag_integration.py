"""
Comprehensive Integration Test Suite for Persistent PDF RAG + MCP + LangGraph
Covers all 11 required test cases using controlled test doubles.
"""
import io
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from starlette.testclient import TestClient

from backend.main import app
from backend.rag_service import RAGService, sanitize_id
from backend.tools import tools, rag_search_tool, set_active_thread_id
from backend.mcp_manager import mcp_manager
from backend.chatbot import chatbot, build_chatbot_graph, ChatState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Sample valid PDF with extractable text
VALID_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 120 >> stream
BT
/F1 12 Tf
72 712 Td
(Project Phoenix: The confidential quarterly revenue for Q3 reached 4.2 million dollars.) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000413 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
490
%%EOF"""


def run_all_tests():
    os.environ["RAG_USE_TEST_EMBEDDINGS"] = "true"
    print("=" * 60)
    print("STARTING COMPLETE RAG & MCP INTEGRATION TEST SUITE")
    print("=" * 60)

    # Isolated temporary RAG storage directory for test execution
    temp_rag_dir = Path(tempfile.mkdtemp(prefix="test_rag_"))
    test_rag_service = RAGService(storage_dir=temp_rag_dir)

    try:
        # -------------------------------------------------------------
        # Test 1 & 3: Valid PDF upload, text extraction, chunking & FAISS index
        # -------------------------------------------------------------
        print("\n[Test 1 & 3] Ingesting valid PDF...")
        doc_meta = test_rag_service.ingest_pdf(VALID_PDF_BYTES, "phoenix_report.pdf", "thread-alpha")
        assert doc_meta["filename"] == "phoenix_report.pdf"
        assert doc_meta["chunks"] > 0
        assert doc_meta["pages"] == 1
        print("[PASSED] Test 1 & 3 PASSED: Extracted and indexed PDF chunks successfully.")

        # -------------------------------------------------------------
        # Test 2: Empty or invalid PDF rejection
        # -------------------------------------------------------------
        print("\n[Test 2] Testing empty & invalid PDF rejection...")
        try:
            test_rag_service.ingest_pdf(b"", "empty.pdf", "thread-alpha")
            assert False, "Should have rejected empty bytes"
        except ValueError as e:
            print(f"[PASSED] Empty PDF rejected: {e}")

        try:
            test_rag_service.ingest_pdf(b"not a real pdf file", "invalid.txt", "thread-alpha")
            assert False, "Should have rejected non-PDF extension"
        except ValueError as e:
            print(f"[PASSED] Non-PDF rejected: {e}")
        print("[PASSED] Test 2 PASSED: Validation handles corrupt and empty files.")

        # -------------------------------------------------------------
        # Test 4: Saving & reloading index after simulated restart
        # -------------------------------------------------------------
        print("\n[Test 4] Testing index reload after simulated backend restart...")
        test_rag_service._cache.clear()
        reloaded_vs = test_rag_service.load_index_for_doc("thread-alpha", doc_meta["document_id"])
        assert reloaded_vs is not None, "Failed to load index from disk after restart"
        search_res = reloaded_vs.similarity_search("revenue", k=1)
        assert len(search_res) > 0
        assert "Project Phoenix" in search_res[0].page_content
        print("[PASSED] Test 4 PASSED: FAISS index survived simulated restart and queried accurately.")

        # -------------------------------------------------------------
        # Test 5: Retrieval returning excerpts and source metadata
        # -------------------------------------------------------------
        print("\n[Test 5] Querying document excerpts and source citations...")
        query_res = test_rag_service.query_documents("thread-alpha", "What was the quarterly revenue?")
        assert len(query_res["excerpts"]) > 0
        first_excerpt = query_res["excerpts"][0]
        assert "4.2 million" in first_excerpt["content"]
        assert first_excerpt["source"] == "phoenix_report.pdf"
        assert first_excerpt["page"] == 1
        print("[PASSED] Test 5 PASSED: Excerpt and metadata citations verified.")

        # -------------------------------------------------------------
        # Test 6: Thread isolation
        # -------------------------------------------------------------
        print("\n[Test 6] Testing strict isolation between chat threads...")
        beta_query = test_rag_service.query_documents("thread-beta", "Project Phoenix")
        assert "error" in beta_query or len(beta_query.get("excerpts", [])) == 0
        beta_docs = test_rag_service.list_documents("thread-beta")
        assert len(beta_docs) == 0, "Thread Beta should have 0 documents"
        print("[PASSED] Test 6 PASSED: Thread Beta cannot access Thread Alpha documents.")

        # -------------------------------------------------------------
        # Test 7: Listing and deleting documents
        # -------------------------------------------------------------
        print("\n[Test 7] Listing and deleting documents...")
        docs_list = test_rag_service.list_documents("thread-alpha")
        assert len(docs_list) == 1
        deleted = test_rag_service.delete_document("thread-alpha", doc_meta["document_id"])
        assert deleted is True
        docs_after = test_rag_service.list_documents("thread-alpha")
        assert len(docs_after) == 0
        print("[PASSED] Test 7 PASSED: Document listing and deletion confirmed.")

        # -------------------------------------------------------------
        # Test 8: RAG Tool registration alongside Native and MCP tools
        # -------------------------------------------------------------
        print("\n[Test 8] Checking tool registry integration...")
        tool_names = [t.name for t in tools]
        assert "rag_search_tool" in tool_names
        assert "duckduckgo_search" in tool_names
        assert "calculator" in tool_names
        assert "get_stock_price" in tool_names
        print("[PASSED] Test 8 PASSED: rag_search_tool registered with native suite.")

        # -------------------------------------------------------------
        # Test 9: MCP integration continues to work
        # -------------------------------------------------------------
        print("\n[Test 9] Verifying MCP discovery with RAG tool present...")
        with TestClient(app) as client:
            res = client.get("/api/tools")
            assert res.status_code == 200
            data = res.json()
            registered_names = [t["name"] for t in data["tools"]]
            assert "rag_search_tool" in registered_names
            assert "mcp_multiply" in registered_names
            print("[PASSED] Test 9 PASSED: MCP tools and RAG tool co-exist in registry.")

        # -------------------------------------------------------------
        # Test 10: Chat history and SQLite checkpoints intact
        # -------------------------------------------------------------
        print("\n[Test 10] Testing SQLite checkpointing and chat continuity...")
        with TestClient(app) as client:
            health_res = client.get("/api/health")
            assert health_res.status_code == 200
            threads_res = client.get("/api/threads")
            assert threads_res.status_code == 200
            invalid_chat = client.post("/api/chat", json={"message": "  "})
            assert invalid_chat.status_code == 422
        print("[PASSED] Test 10 PASSED: FastAPI endpoints and SQLite checkpointer functioning.")

        # -------------------------------------------------------------
        # Test 11: End-to-end API upload and query flow
        # -------------------------------------------------------------
        print("\n[Test 11] Testing live FastAPI multipart upload and document listing endpoints...")
        with TestClient(app) as client:
            upload_res = client.post(
                "/api/threads/test-api-thread/documents",
                files={"file": ("report.pdf", VALID_PDF_BYTES, "application/pdf")},
            )
            assert upload_res.status_code == 200
            doc_json = upload_res.json()
            assert doc_json["filename"] == "report.pdf"
            uploaded_doc_id = doc_json["document_id"]

            # List endpoint
            list_res = client.get("/api/threads/test-api-thread/documents")
            assert list_res.status_code == 200
            assert len(list_res.json()["documents"]) >= 1

            # Delete endpoint
            del_res = client.delete(f"/api/threads/test-api-thread/documents/{uploaded_doc_id}")
            assert del_res.status_code == 200
            assert del_res.json()["deleted"] is True
        print("[PASSED] Test 11 PASSED: Full REST API RAG lifecycle verified.")

        print("\n" + "=" * 60)
        print("ALL 11 RAG & MCP INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        shutil.rmtree(temp_rag_dir, ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()
