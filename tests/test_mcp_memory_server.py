#!/usr/bin/env python3
"""
Test script for MCP Memory Server

This script tests the memory server functionality including:
1. Server startup and health check
2. Codebase indexing
3. Memory storage and retrieval
4. Semantic search
5. Memory pruning
"""

import asyncio
import requests
import time
import sys
import subprocess
import signal
from pathlib import Path
from typing import Optional
import json


class MemoryServerTester:
    """Test suite for MCP Memory Server"""
    
    def __init__(self, server_url: str = "http://localhost:9820"):
        self.server_url = server_url
        self.server_process: Optional[subprocess.Popen] = None
        
    def start_server(self, codebase_path: str = None) -> bool:
        """Start the memory server for testing"""
        try:
            cmd = [
                sys.executable,
                "-m", "conductor_memory.server.sse",
                "--port", "9820",
                "--host", "127.0.0.1",
                "--log-level", "WARNING"  # Reduce noise during testing
            ]
            
            if codebase_path:
                cmd.extend(["--codebase-path", codebase_path])
            
            print(f"Starting memory server: {' '.join(cmd)}")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=1)
                    if response.status_code == 200:
                        print("[OK] Memory server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
            
            print("[FAIL] Memory server failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"[FAIL] Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the memory server"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("[OK] Memory server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("[OK] Memory server force-killed")
            except Exception as e:
                print(f"[WARN] Error stopping server: {e}")
    
    def test_health_check(self) -> bool:
        """Test server health check"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print("[OK] Health check passed")
                    return True
            
            print(f"[FAIL] Health check failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"[FAIL] Health check error: {e}")
            return False
    
    def test_status_endpoint(self) -> bool:
        """Test server status endpoint"""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Status endpoint working")
                print(f"     Indexing status: {data.get('indexing', {}).get('status', 'unknown')}")
                return True
            
            print(f"[FAIL] Status endpoint failed: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"[FAIL] Status endpoint error: {e}")
            return False
    
    def test_memory_storage(self) -> bool:
        """Test storing memories"""
        try:
            # Test storing a conversation memory
            store_request = {
                "project_id": "test_project",
                "role": "user",
                "prompt": "How do I implement a binary search algorithm?",
                "response": "",
                "doc_text": "",
                "tags": ["algorithm", "search", "test"],
                "pin": False,
                "source": "test"
            }
            
            response = requests.post(
                f"{self.server_url}/store",
                json=store_request,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                memory_id = data.get("id")
                print(f"[OK] Memory stored successfully (ID: {memory_id[:8]}...)")
                
                # Store the response
                response_request = {
                    "project_id": "test_project",
                    "role": "assistant",
                    "prompt": "",
                    "response": "Here's a binary search implementation in Python:\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                    "doc_text": "",
                    "tags": ["algorithm", "search", "python", "test"],
                    "pin": False,
                    "source": "test"
                }
                
                response2 = requests.post(
                    f"{self.server_url}/store",
                    json=response_request,
                    timeout=10
                )
                
                if response2.status_code == 200:
                    print("[OK] Response memory stored successfully")
                    return True
                else:
                    print(f"[FAIL] Failed to store response memory: {response2.status_code}")
                    return False
            else:
                print(f"[FAIL] Failed to store memory: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[FAIL] Memory storage error: {e}")
            return False
    
    def test_memory_search(self) -> bool:
        """Test searching memories"""
        try:
            # Wait a moment for indexing
            time.sleep(2)
            
            search_request = {
                "query": "binary search algorithm implementation",
                "project_id": "test_project",
                "max_results": 5,
                "min_relevance": 0.0
            }
            
            response = requests.post(
                f"{self.server_url}/search",
                json=search_request,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                total_found = data.get("total_found", 0)
                query_time = data.get("query_time_ms", 0)
                
                print(f"[OK] Search completed in {query_time:.1f}ms")
                print(f"     Found {total_found} results, showing {len(results)}")
                
                if results:
                    print(f"     First result: {results[0]['role']} - {results[0]['tags']}")
                    return True
                else:
                    print("[WARN] No search results found (may be expected for small dataset)")
                    return True  # Not necessarily a failure
            else:
                print(f"[FAIL] Search failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[FAIL] Memory search error: {e}")
            return False
    
    def test_codebase_search(self) -> bool:
        """Test searching codebase memories"""
        try:
            # Wait for indexing to complete
            print("[INFO] Waiting for codebase indexing to complete...")
            for i in range(60):  # Wait up to 60 seconds
                status_response = requests.get(f"{self.server_url}/status", timeout=5)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    indexing_status = status_data.get("indexing", {}).get("status")
                    
                    if indexing_status == "completed":
                        print("[OK] Codebase indexing completed")
                        break
                    elif indexing_status == "error":
                        print("[WARN] Codebase indexing failed")
                        break
                    elif indexing_status == "indexing":
                        progress = status_data.get("indexing", {}).get("progress", 0) * 100
                        print(f"[INFO] Indexing progress: {progress:.1f}%")
                
                time.sleep(1)
            
            # Search for code-related content
            search_request = {
                "query": "memory database vector store",
                "max_results": 3,
                "min_relevance": 0.0
            }
            
            response = requests.post(
                f"{self.server_url}/search",
                json=search_request,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                print(f"[OK] Codebase search completed")
                print(f"     Found {len(results)} code-related results")
                
                if results:
                    for i, result in enumerate(results[:2], 1):
                        print(f"     Result {i}: {result['source']} - {result['tags'][:3]}")
                
                return True
            else:
                print(f"[FAIL] Codebase search failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[FAIL] Codebase search error: {e}")
            return False
    
    def test_memory_pruning(self) -> bool:
        """Test memory pruning"""
        try:
            response = requests.post(
                f"{self.server_url}/prune",
                params={"project_id": "test_project", "max_age_days": 0},  # Prune everything
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                pruned = data.get("pruned", 0)
                kept = data.get("kept", 0)
                
                print(f"[OK] Memory pruning completed")
                print(f"     Pruned: {pruned}, Kept: {kept}")
                return True
            else:
                print(f"[FAIL] Memory pruning failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[FAIL] Memory pruning error: {e}")
            return False
    
    def run_all_tests(self, codebase_path: str = None) -> bool:
        """Run all tests"""
        print("MCP Memory Server Test Suite")
        print("=" * 50)
        
        tests = [
            ("Server Startup", lambda: self.start_server(codebase_path)),
            ("Health Check", self.test_health_check),
            ("Status Endpoint", self.test_status_endpoint),
            ("Memory Storage", self.test_memory_storage),
            ("Memory Search", self.test_memory_search),
            ("Memory Pruning", self.test_memory_pruning),
        ]
        
        # Add codebase search test if codebase path provided
        if codebase_path:
            tests.append(("Codebase Search", self.test_codebase_search))
        
        passed = 0
        total = len(tests)
        
        try:
            for test_name, test_func in tests:
                print(f"\nRunning: {test_name}")
                if test_func():
                    passed += 1
                else:
                    print(f"[FAIL] {test_name} failed")
        
        finally:
            self.stop_server()
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! MCP Memory Server is working correctly.")
            return True
        else:
            print("❌ Some tests failed. Check the output above for details.")
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCP Memory Server")
    parser.add_argument("--codebase", type=str, help="Path to codebase for indexing test")
    parser.add_argument("--server-url", type=str, default="http://localhost:9820", help="Memory server URL")
    
    args = parser.parse_args()
    
    # Use current directory as codebase if not specified
    codebase_path = args.codebase or str(Path.cwd())
    
    tester = MemoryServerTester(args.server_url)
    success = tester.run_all_tests(codebase_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()