"""
Reference local MCP Math Server.
Provides basic arithmetic utilities over standard I/O (stdio) transport.
Configurable via MCP_MATH_SERVER_PATH and MCP_MATH_SERVER_COMMAND.
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")


@mcp.tool()
def mcp_add(a: float, b: float) -> float:
    """Add two numbers using the MCP math server."""
    return a + b


@mcp.tool()
def mcp_multiply(a: float, b: float) -> float:
    """Multiply two numbers using the MCP math server."""
    return a * b


@mcp.tool()
def mcp_power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent using MCP math server."""
    return base ** exponent


if __name__ == "__main__":
    mcp.run()
