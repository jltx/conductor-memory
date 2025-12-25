#!/usr/bin/env python3
"""
Test script to queue a codebase for summarization via MCP tool.
"""
import requests
import json

def queue_codebase(codebase_name: str, only_missing: bool = True):
    """Queue a codebase for summarization via MCP tool."""
    
    # MCP tool call format for SSE server
    url = "http://localhost:9820/mcp/call"
    
    payload = {
        "tool": "memory_queue_codebase_summarization",
        "arguments": {
            "codebase": codebase_name,
            "only_missing": only_missing
        }
    }
    
    print(f"Queuing codebase '{codebase_name}' for summarization...")
    print(f"Only missing: {only_missing}")
    print()
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("Result:")
        print(json.dumps(result, indent=2))
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling MCP tool: {e}")
        return None


def check_summarization_status():
    """Check current summarization status."""
    url = "http://localhost:9820/api/summarization"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        status = response.json()
        
        print("\n=== Summarization Status ===")
        print(f"Enabled: {status.get('enabled')}")
        print(f"LLM Enabled: {status.get('llm_enabled')}")
        print(f"Running: {status.get('is_running')}")
        print(f"Files Queued: {status.get('files_queued')}")
        print(f"Files Completed: {status.get('files_completed')}")
        print(f"Files Skipped: {status.get('files_skipped')}")
        print(f"Current File: {status.get('current_file')}")
        print()
        
        # Show by-codebase stats
        by_codebase = status.get('by_codebase', {})
        for codebase_name, stats in by_codebase.items():
            total = stats.get('total_summarized', 0)
            print(f"  {codebase_name}: {total} files summarized")
        
        return status
        
    except requests.exceptions.RequestException as e:
        print(f"Error checking status: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    # Check current status first
    print("Checking current summarization status...\n")
    check_summarization_status()
    
    # Queue the new codebase
    codebase = sys.argv[1] if len(sys.argv) > 1 else "truthsocial-android"
    only_missing = True if len(sys.argv) <= 2 else sys.argv[2].lower() == "true"
    
    result = queue_codebase(codebase, only_missing)
    
    if result and result.get("success"):
        print("\n✅ Successfully queued codebase for summarization!")
        print("\nWait a moment, then check status again...\n")
        
        import time
        time.sleep(2)
        
        check_summarization_status()
    else:
        print("\n❌ Failed to queue codebase")
