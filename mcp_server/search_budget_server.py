import re
from typing import List, Dict, Any

    
from fastmcp import FastMCP
import re
import json
from typing import List, Dict, Any

app = FastMCP("search-budget-server")

@app.tool("search_budget_text")
def search_budget_text(keyword: str, structured_json_path: str) -> List[Dict[str, Any]]:
    """
    Search parsed budget text JSON for a given keyword (one word). Keyword example is revenue, expenditure, spending etc.
    Args:
        keyword (str): Keyword to search.
        structured_json_path (str): Path to JSON containing budget elements.
    """
    with open(structured_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elements = data.get("elements", [])
    results = []
    for c in elements:
        text = c.get("content_markdown", "")
        if re.search(keyword, text, re.IGNORECASE):
            results.append({"page": c.get("page"), "text": text.strip()})
    return results

if __name__ == "__main__":
    app.run()