"""Test storing via SSE server and checking persistence"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import requests
import uuid
import time

SSE_URL = "http://127.0.0.1:9820"

def test_store_via_sse():
    """Call the MCP store tool via the SSE server's JSON-RPC interface"""
    
    # First, let's see what endpoints are available
    print("=== Testing SSE Server ===")
    
    # Try the tools/list endpoint
    try:
        resp = requests.get(f"{SSE_URL}/")
        print(f"Root response: {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
    # The MCP SSE protocol uses JSON-RPC over SSE
    # Let's try to call the memory_store tool directly
    
    # Create a unique test ID
    test_id = str(uuid.uuid4())[:8]
    test_content = f"SSE TEST {test_id}: This is a test stored via SSE at {time.time()}"
    
    print(f"\nTest content: {test_content}")
    
    # MCP JSON-RPC request format
    rpc_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "memory_store",
            "arguments": {
                "content": test_content,
                "tags": ["test", "sse-test"],
                "source": "sse_test_script"
            }
        }
    }
    
    print(f"\nSending JSON-RPC request...")
    print(json.dumps(rpc_request, indent=2))
    
    try:
        # Try POST to /messages endpoint (common MCP endpoint)
        resp = requests.post(
            f"{SSE_URL}/messages",
            json=rpc_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nResponse status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_store_via_sse()
