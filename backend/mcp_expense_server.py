"""
Local MCP Expense Server.
Provides expense management tools (add_expense, list_expenses, summarize)
over standard I/O (stdio) transport using FastMCP.
"""
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LocalExpenseServer")

# In-memory expense records with initial sample data
_EXPENSES: List[Dict[str, Any]] = [
    {"id": 1, "description": "Coffee and snacks", "amount": 8.50, "category": "Food"},
    {"id": 2, "description": "Cloud server subscription", "amount": 29.99, "category": "Technology"},
    {"id": 3, "description": "AI & Machine Learning Book", "amount": 45.00, "category": "Education"},
]


@mcp.tool()
def add_expense(description: str, amount: float, category: str = "general") -> dict:
    """
    Record a new expense.
    Parameters:
      description: What was purchased
      amount: Cost of the expense in dollars
      category: Classification (e.g., Food, Tech, Travel, Office)
    """
    new_id = len(_EXPENSES) + 1
    record = {
        "id": new_id,
        "description": description,
        "amount": round(float(amount), 2),
        "category": category,
    }
    _EXPENSES.append(record)
    return {
        "status": "success",
        "message": f"Added expense #{new_id}: {description} (${record['amount']})",
        "expense": record,
    }


@mcp.tool()
def list_expenses() -> list:
    """Retrieve all logged expense records."""
    return _EXPENSES


@mcp.tool()
def summarize() -> dict:
    """Summarize all recorded expenses with totals and category breakdown."""
    total = sum(e["amount"] for e in _EXPENSES)
    breakdown: Dict[str, float] = {}
    for e in _EXPENSES:
        cat = e.get("category", "general")
        breakdown[cat] = round(breakdown.get(cat, 0.0) + e["amount"], 2)

    return {
        "total_amount": round(total, 2),
        "total_count": len(_EXPENSES),
        "category_breakdown": breakdown,
    }


if __name__ == "__main__":
    mcp.run()
