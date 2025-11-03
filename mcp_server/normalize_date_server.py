from fastmcp import FastMCP
from datetime import datetime
import re

app = FastMCP("normalize-date-server")

@app.tool("normalize_date")
def normalize_date(date_string: str) -> str:
    """
    Normalize budget-style dates to ISO YYYY-MM-DD.
    Examples:
    - "16 February 2024" -> "2024-02-16"
    - "1 Jan 2024" -> "2024-01-01"
    """
    date_string = (date_string or "").strip()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(date_string, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Search with regex and normalize        
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", date_string)
    if m:
        d, month, y = m.groups()
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(f"{d} {month} {y}", fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return ""

if __name__ == "__main__":
    app.run()