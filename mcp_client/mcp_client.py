import time
import json
import subprocess
from typing import Any, Dict, Optional

class MCPClient:
    """Handles JSON-RPC communication with the FastMCP server."""

    def __init__(self, server_path: str):
        self.server_path = server_path

    def _read_until_result(self, proc, target_id: int, timeout=3.0):
        start = time.time()
        buffer = ""
        while time.time() - start < timeout:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            buffer += line
            try:
                msg = json.loads(line)
                if msg.get("id") == target_id:
                    return msg
            except json.JSONDecodeError:
                continue
        raise TimeoutError(f"Timed out waiting for MCP response.\nPartial buffer:\n{buffer}")

    def call(
        self,
        method_name_or_arg: Any,
        arguments: Optional[Dict[str, Any]] = None,
        default_method: str = "normalize_date",
    ) -> Any:
        
        if arguments is None and isinstance(method_name_or_arg, str):
            method_name = default_method
            arguments = {"date_string": method_name_or_arg}
        else:
            method_name = method_name_or_arg

        proc = subprocess.Popen(
            ["python", self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # === Handshake ===
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "LangGraphBudgetPipeline", "version": "0.1"},
            },
        }
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()
        self._read_until_result(proc, 0)

        # === Call tool ===
        call_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": method_name, "arguments": arguments},
        }
        proc.stdin.write(json.dumps(call_request) + "\n")
        proc.stdin.flush()
        response = self._read_until_result(proc, 1)

        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()

        result = response.get("result", {})
        content = result.get("content", [])
        if isinstance(content, list) and content and "text" in content[0]:
            return content[0]["text"]
        return result.get("content", [])