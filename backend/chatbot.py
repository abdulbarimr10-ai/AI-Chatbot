import os
import sqlite3
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

import logging
from backend.tools import tools as native_tools, set_active_thread_id, set_active_openai_key
from backend.mcp_manager import mcp_manager

logger = logging.getLogger(__name__)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "backend" / ".env")

# Determine consistent absolute path for SQLite database
DB_PATH = os.getenv("CHATBOT_DB_PATH", str(BASE_DIR / "chatbot.db"))


# 1. LLM initialization (Supports ChatOpenAI with automatic fallback to ChatHuggingFace and per-request user keys)
def create_llm_instance(openai_key: Optional[str] = None, hf_token: Optional[str] = None):
    effective_hf = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    effective_openai = openai_key or os.getenv("OPENAI_API_KEY")

    if hf_token or (effective_hf and (not effective_openai or effective_openai == "Gareebon")):
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            repo_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
            provider = os.getenv("HF_PROVIDER", "featherless-ai")
            endpoint = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                provider=provider,
                max_new_tokens=256,
                huggingfacehub_api_token=effective_hf,
            )
            return ChatHuggingFace(llm=endpoint)
        except Exception as e:
            logger.warning(f"Failed to initialize ChatHuggingFace: {e}")

    from langchain_openai import ChatOpenAI
    if effective_openai and not effective_openai.startswith("sk-ijklmnop") and effective_openai != "Gareebon":
        return ChatOpenAI(api_key=effective_openai)
    return ChatOpenAI()


llm = create_llm_instance()


# 2. State definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Global graph and tool state
current_tools: List[BaseTool] = list(native_tools)
llm_with_tools = llm.bind_tools(current_tools) if hasattr(llm, "bind_tools") else llm

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an intelligent, helpful AI assistant with access to tools:\n"
        "- Web Search (`duckduckgo_search`)\n"
        "- Arithmetic Calculator (`calculator`)\n"
        "- Stock Market Quotes (`get_stock_price`)\n"
        "- Document Retrieval (`rag_search_tool`): Use this when the user asks questions about "
        "their uploaded PDF documents. Always cite source filenames and page numbers in your answers. "
        "Distinguish verified evidence from unsupported claims. If the document does not contain "
        "the answer, honestly communicate that limitation.\n"
        "- Model Context Protocol (MCP) Tools: Execute any connected MCP tools when relevant."
    )
)


# 3. Graph Nodes
def chat_node(state: ChatState, config: Optional[RunnableConfig] = None):
    """LLM node that either answers directly or invokes tool calls, supporting per-turn API keys."""
    configurable = (config or {}).get("configurable", {})
    user_openai_key = configurable.get("openai_api_key")
    user_hf_token = configurable.get("hf_token")

    if user_openai_key or user_hf_token:
        # Dynamically instantiate model with user-provided key
        turn_llm = create_llm_instance(openai_key=user_openai_key, hf_token=user_hf_token)
        turn_llm_with_tools = turn_llm.bind_tools(current_tools) if hasattr(turn_llm, "bind_tools") and current_tools else turn_llm
    else:
        turn_llm_with_tools = llm_with_tools

    messages = [SYSTEM_PROMPT, *state["messages"]]
    response = turn_llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 4. Checkpointer setup
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# 5. Build Graph function (allows dynamic registration of discovered MCP tools & RAG tool)
def build_chatbot_graph(tools_to_register: Optional[List[BaseTool]] = None):
    global chatbot, llm_with_tools, current_tools

    active_tools = tools_to_register if tools_to_register is not None else list(native_tools)
    current_tools = active_tools
    llm_with_tools = llm.bind_tools(active_tools) if hasattr(llm, "bind_tools") else llm

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(active_tools))

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot


# Initialize baseline graph with native tools
chatbot = build_chatbot_graph(native_tools)


# 6. Thread & History Helper Functions
def retrieve_all_threads() -> List[str]:
    """Retrieve all distinct thread IDs stored in SQLite checkpoints."""
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                all_threads.add(str(thread_id))
    except Exception:
        return []
    return sorted(list(all_threads), reverse=True)


def get_thread_messages(thread_id: str) -> List[Dict[str, str]]:
    """Retrieve formatted message history for a given thread ID."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config=config)
    messages = state.values.get("messages", []) if state and state.values else []

    formatted_messages: List[Dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            formatted_messages.append({"role": "assistant", "content": str(msg.content)})

    return formatted_messages


def execute_chat(
    user_message: str,
    thread_id: str,
    openai_api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Execute the LangGraph workflow for a single chat turn and return the final AI text."""
    set_active_thread_id(thread_id)
    if openai_api_key and openai_api_key.strip():
        set_active_openai_key(openai_api_key.strip())

    configurable = {"thread_id": thread_id}
    if openai_api_key and openai_api_key.strip():
        configurable["openai_api_key"] = openai_api_key.strip()
    if hf_token and hf_token.strip():
        configurable["hf_token"] = hf_token.strip()

    config = {
        "configurable": configurable,
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    input_payload = {"messages": [HumanMessage(content=user_message)]}
    result = chatbot.invoke(input_payload, config=config)

    # Extract the last message from the assistant
    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        return getattr(last_msg, "content", "")
    return ""
