import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from backend.tools import tools as native_tools

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def make_sync_compatible_tool(tool: BaseTool, server_name: str) -> BaseTool:
    """
    Wrap an asynchronous MCP tool so it can be executed safely in both
    asynchronous and synchronous execution contexts (including LangGraph ToolNode).
    """
    original_coroutine = tool.coroutine

    def sync_runner(**kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If called from within an active event loop thread, delegate to a worker thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(tool.ainvoke(kwargs))).result()
        else:
            return asyncio.run(tool.ainvoke(kwargs))

    wrapped_tool = StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=sync_runner,
        coroutine=original_coroutine,
        metadata={
            "source": "mcp",
            "server": server_name,
        },
    )
    return wrapped_tool


class MCPManager:
    """
    Modular manager responsible for MCP server discovery, connection management,
    lifecycle handling, and tool registry synchronization.
    """

    def __init__(self):
        self.discovered_mcp_tools: List[BaseTool] = []
        self.server_statuses: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    def get_configured_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Build MCP server connection dictionaries from environment configuration.
        Replaces hardcoded local machine paths with configurable cross-platform settings.
        """
        servers: Dict[str, Dict[str, Any]] = {}

        # 1. Math MCP Server (Local stdio transport)
        math_enabled = os.getenv("MCP_MATH_SERVER_ENABLED", "true").lower() in ("true", "1", "yes")
        math_command = os.getenv("MCP_MATH_SERVER_COMMAND", sys.executable)
        default_math_path = str(BASE_DIR / "backend" / "mcp_math_server.py")
        math_path = os.getenv("MCP_MATH_SERVER_PATH", default_math_path)

        if math_enabled:
            if os.path.exists(math_path):
                servers["math"] = {
                    "transport": "stdio",
                    "command": math_command,
                    "args": [math_path],
                }
            else:
                self.server_statuses["math"] = {
                    "name": "math",
                    "transport": "stdio",
                    "connected": False,
                    "tools": [],
                    "error": f"Math server script not found at '{math_path}'",
                }

        # 2. Expense MCP Server (Local stdio by default, with remote fallback)
        expense_enabled = os.getenv("MCP_EXPENSE_SERVER_ENABLED", "true").lower() in ("true", "1", "yes")
        expense_command = os.getenv("MCP_EXPENSE_SERVER_COMMAND", sys.executable)
        default_expense_path = str(BASE_DIR / "backend" / "mcp_expense_server.py")
        expense_path = os.getenv("MCP_EXPENSE_SERVER_PATH", default_expense_path)
        expense_url = os.getenv("MCP_EXPENSE_SERVER_URL")

        if expense_enabled:
            if expense_url:
                # Use remote HTTP/SSE if explicitly provided in environment
                expense_transport = os.getenv("MCP_EXPENSE_SERVER_TRANSPORT", "streamable_http")
                servers["expense"] = {
                    "transport": expense_transport,
                    "url": expense_url,
                }
            elif os.path.exists(expense_path):
                # Use local stdio expense server
                servers["expense"] = {
                    "transport": "stdio",
                    "command": expense_command,
                    "args": [expense_path],
                }
            else:
                self.server_statuses["expense"] = {
                    "name": "expense",
                    "transport": "stdio",
                    "connected": False,
                    "tools": [],
                    "error": f"Expense server script not found at '{expense_path}'",
                }

        return servers

    async def initialize(self) -> List[BaseTool]:
        """
        Connect to each configured MCP server, discover its tools,
        and handle failures gracefully without crashing the core chatbot.
        """
        configured_servers = self.get_configured_servers()
        all_discovered: List[BaseTool] = []

        for server_name, conn_params in configured_servers.items():
            try:
                logger.info(f"Connecting to MCP server '{server_name}'...")
                client = MultiServerMCPClient({server_name: conn_params})
                
                # Discover tools with a 10s connection timeout
                raw_tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)

                compatible_tools = [
                    make_sync_compatible_tool(t, server_name)
                    for t in raw_tools
                ]
                all_discovered.extend(compatible_tools)

                self.server_statuses[server_name] = {
                    "name": server_name,
                    "transport": conn_params.get("transport"),
                    "connected": True,
                    "tools": [t.name for t in compatible_tools],
                    "error": None,
                }
                logger.info(f"MCP server '{server_name}' connected: {len(compatible_tools)} tools discovered.")
            except Exception as exc:
                err_msg = str(exc) or type(exc).__name__
                logger.warning(f"MCP server '{server_name}' discovery failed: {err_msg}")
                self.server_statuses[server_name] = {
                    "name": server_name,
                    "transport": conn_params.get("transport"),
                    "connected": False,
                    "tools": [],
                    "error": f"Connection failed: {err_msg}",
                }

        self.discovered_mcp_tools = all_discovered
        self._initialized = True
        return all_discovered

    def get_all_tools(self) -> List[BaseTool]:
        """
        Return the combined registry of native tools and active MCP tools.
        """
        return list(native_tools) + self.discovered_mcp_tools

    def get_status_report(self) -> Dict[str, Any]:
        """
        Return detailed status of native and MCP tools for the GET /api/tools endpoint.
        """
        tool_list = []

        # Native tools
        for t in native_tools:
            tool_list.append({
                "name": t.name,
                "description": t.description,
                "type": "native",
                "server": "internal",
                "available": True,
            })

        # MCP tools
        for t in self.discovered_mcp_tools:
            meta = getattr(t, "metadata", {}) or {}
            tool_list.append({
                "name": t.name,
                "description": t.description,
                "type": "mcp",
                "server": meta.get("server", "unknown"),
                "available": True,
            })

        return {
            "total_tools": len(tool_list),
            "native_tools_count": len(native_tools),
            "mcp_tools_count": len(self.discovered_mcp_tools),
            "servers": self.server_statuses,
            "tools": tool_list,
        }

    async def cleanup(self):
        """Clean up active MCP connections on application shutdown."""
        self.discovered_mcp_tools.clear()
        self.server_statuses.clear()
        self._initialized = False
        logger.info("MCP Manager resources released.")


# Singleton instance
mcp_manager = MCPManager()
