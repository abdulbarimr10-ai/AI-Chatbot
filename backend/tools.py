import os
import contextvars
import requests
from typing import Optional
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# ContextVar for thread-safe active thread tracking across async/sync executions
_active_thread_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("active_thread_id", default=None)
_active_openai_key: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("active_openai_key", default=None)


def set_active_thread_id(thread_id: Optional[str]):
    """Set active thread ID in context for tool execution."""
    return _active_thread_id.set(thread_id)


def get_active_thread_id() -> Optional[str]:
    """Get active thread ID from context."""
    return _active_thread_id.get()


def set_active_openai_key(key: Optional[str]):
    """Set active user-provided OpenAI key in context."""
    return _active_openai_key.set(key)


def get_active_openai_key() -> Optional[str]:
    """Get active user-provided OpenAI key from context."""
    return _active_openai_key.get()


# 1. DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun(region="us-en")


# 2. Arithmetic Calculator Tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


# 3. Stock Price Tool with secure environment configuration and timeout
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY environment variable is not configured. Please set it in backend/.env"
        }

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve stock quote: {str(e)}"}


# 4. Persistent RAG Retrieval Tool
@tool
def rag_search_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Search and retrieve relevant text excerpts from uploaded PDF documents associated with this chat thread.
    Use this tool whenever the user asks questions about their uploaded files, PDFs, reports, or documents.
    Always cite the source filename and page number from the returned excerpts.
    """
    effective_tid = thread_id or get_active_thread_id()
    if not effective_tid:
        return {
            "error": "No active conversation thread ID found to query uploaded documents.",
            "query": query,
            "excerpts": [],
        }

    from backend.rag_service import rag_service
    return rag_service.query_documents(effective_tid, query)


# Consolidated list of native tools
tools = [search_tool, get_stock_price, calculator, rag_search_tool]
