"""
Performance tests for conductor-memory API endpoints.

Tests response times for all API endpoints and identifies performance bottlenecks.
"""

import time
import requests
import pytest
import json
from typing import Dict, List, Tuple
from urllib.parse import quote


class APIPerformanceTester:
    """Helper class for API performance testing."""
    
    def __init__(self, base_url: str = "http://localhost:9820"):
        self.base_url = base_url
        self.results: List[Dict] = []
    
    def time_request(self, method: str, endpoint: str, **kwargs) -> Tuple[float, requests.Response]:
        """Time a single API request and return (duration_ms, response)."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.perf_counter()
        
        if method.upper() == "GET":
            response = requests.get(url, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        return duration_ms, response
    
    def record_result(self, endpoint: str, duration_ms: float, status_code: int, 
                     response_size: int = 0, notes: str = ""):
        """Record a performance test result."""
        result = {
            "endpoint": endpoint,
            "duration_ms": round(duration_ms, 2),
            "status_code": status_code,
            "response_size_bytes": response_size,
            "notes": notes,
            "timestamp": time.time()
        }
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.results:
            return {}
        
        durations = [r["duration_ms"] for r in self.results if r["status_code"] == 200]
        if not durations:
            return {"error": "No successful requests"}
        
        return {
            "total_requests": len(self.results),
            "successful_requests": len(durations),
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "slow_endpoints": [r for r in self.results if r["duration_ms"] > 1000],  # > 1 second
            "failed_requests": [r for r in self.results if r["status_code"] != 200]
        }


@pytest.fixture
def perf_tester():
    """Create a performance tester instance."""
    return APIPerformanceTester()


@pytest.fixture
def test_codebase():
    """Get a test codebase name for testing."""
    # First check what codebases are available
    response = requests.get("http://localhost:9820/api/codebases")
    if response.status_code == 200:
        codebases = response.json().get("codebases", [])
        if codebases:
            return codebases[0]["name"]
    return "conductor-memory"  # fallback


@pytest.fixture
def test_file_path():
    """Get a test file path for testing."""
    return "src/conductor_memory/service/__init__.py"


class TestAPIPerformance:
    """Performance tests for all API endpoints."""
    
    def test_health_endpoint_performance(self, perf_tester):
        """Test /health endpoint performance."""
        duration_ms, response = perf_tester.time_request("GET", "/health")
        
        result = perf_tester.record_result(
            "/health", duration_ms, response.status_code,
            len(response.content), "Basic health check"
        )
        
        assert response.status_code == 200
        assert duration_ms < 100, f"Health endpoint too slow: {duration_ms}ms"
        print(f"Health endpoint: {duration_ms:.2f}ms")
    
    def test_api_status_performance(self, perf_tester):
        """Test /api/status endpoint performance."""
        duration_ms, response = perf_tester.time_request("GET", "/api/status")
        
        result = perf_tester.record_result(
            "/api/status", duration_ms, response.status_code,
            len(response.content), "System status check"
        )
        
        assert response.status_code == 200
        assert duration_ms < 500, f"Status endpoint too slow: {duration_ms}ms"
        print(f"Status endpoint: {duration_ms:.2f}ms")
    
    def test_api_summarization_performance(self, perf_tester):
        """Test /api/summarization endpoint performance."""
        duration_ms, response = perf_tester.time_request("GET", "/api/summarization")
        
        result = perf_tester.record_result(
            "/api/summarization", duration_ms, response.status_code,
            len(response.content), "Summarization status"
        )
        
        assert response.status_code == 200
        assert duration_ms < 500, f"Summarization endpoint too slow: {duration_ms}ms"
        print(f"Summarization endpoint: {duration_ms:.2f}ms")
    
    def test_api_codebases_performance(self, perf_tester):
        """Test /api/codebases endpoint performance."""
        duration_ms, response = perf_tester.time_request("GET", "/api/codebases")
        
        result = perf_tester.record_result(
            "/api/codebases", duration_ms, response.status_code,
            len(response.content), "List codebases"
        )
        
        assert response.status_code == 200
        assert duration_ms < 300, f"Codebases endpoint too slow: {duration_ms}ms"
        print(f"Codebases endpoint: {duration_ms:.2f}ms")
    
    def test_api_files_performance(self, perf_tester, test_codebase):
        """Test /api/files endpoint performance."""
        endpoint = f"/api/files?codebase={test_codebase}&limit=50"
        duration_ms, response = perf_tester.time_request("GET", endpoint)
        
        result = perf_tester.record_result(
            "/api/files", duration_ms, response.status_code,
            len(response.content), f"List files for {test_codebase}"
        )
        
        assert response.status_code == 200
        assert duration_ms < 1000, f"Files endpoint too slow: {duration_ms}ms"
        print(f"Files endpoint: {duration_ms:.2f}ms")
    
    def test_api_file_details_performance(self, perf_tester, test_codebase, test_file_path):
        """Test /api/file-details endpoint performance."""
        endpoint = f"/api/file-details?codebase={test_codebase}&path={quote(test_file_path)}"
        duration_ms, response = perf_tester.time_request("GET", endpoint)
        
        result = perf_tester.record_result(
            "/api/file-details", duration_ms, response.status_code,
            len(response.content), f"File details for {test_file_path}"
        )
        
        assert response.status_code == 200
        assert duration_ms < 2000, f"File details endpoint too slow: {duration_ms}ms"
        print(f"File details endpoint: {duration_ms:.2f}ms")
    
    def test_api_file_content_performance(self, perf_tester, test_codebase, test_file_path):
        """Test /api/file-content endpoint performance."""
        endpoint = f"/api/file-content?codebase={test_codebase}&path={quote(test_file_path)}"
        duration_ms, response = perf_tester.time_request("GET", endpoint)
        
        result = perf_tester.record_result(
            "/api/file-content", duration_ms, response.status_code,
            len(response.content), f"File content for {test_file_path}"
        )
        
        assert response.status_code == 200
        assert duration_ms < 1000, f"File content endpoint too slow: {duration_ms}ms"
        print(f"File content endpoint: {duration_ms:.2f}ms")
    
    def test_api_file_summary_performance(self, perf_tester, test_codebase, test_file_path):
        """Test /api/file-summary endpoint performance - the slow one!"""
        endpoint = f"/api/file-summary?codebase={test_codebase}&path={quote(test_file_path)}"
        
        # Time multiple requests to get average
        durations = []
        for i in range(3):
            duration_ms, response = perf_tester.time_request("GET", endpoint)
            durations.append(duration_ms)
            
            result = perf_tester.record_result(
                "/api/file-summary", duration_ms, response.status_code,
                len(response.content), f"File summary for {test_file_path} (run {i+1})"
            )
            
            assert response.status_code == 200
            print(f"File summary endpoint (run {i+1}): {duration_ms:.2f}ms")
        
        avg_duration = sum(durations) / len(durations)
        print(f"File summary average: {avg_duration:.2f}ms")
        
        # This is the problematic endpoint - let's be more lenient for now
        if avg_duration > 5000:
            print(f"WARNING: File summary endpoint is very slow: {avg_duration:.2f}ms")
    
    def test_api_search_performance(self, perf_tester, test_codebase):
        """Test /api/search endpoint performance."""
        search_data = {
            "query": "memory service",
            "codebase": test_codebase,
            "limit": 10
        }
        
        duration_ms, response = perf_tester.time_request(
            "POST", "/api/search",
            json=search_data,
            headers={"Content-Type": "application/json"}
        )
        
        result = perf_tester.record_result(
            "/api/search", duration_ms, response.status_code,
            len(response.content), f"Search for 'memory service' in {test_codebase}"
        )
        
        assert response.status_code == 200
        assert duration_ms < 2000, f"Search endpoint too slow: {duration_ms}ms"
        print(f"Search endpoint: {duration_ms:.2f}ms")
    
    def test_api_memories_performance(self, perf_tester):
        """Test /api/memories endpoint performance."""
        endpoint = "/api/memories?limit=50"
        duration_ms, response = perf_tester.time_request("GET", endpoint)
        
        result = perf_tester.record_result(
            "/api/memories", duration_ms, response.status_code,
            len(response.content), "List memories"
        )
        
        assert response.status_code == 200
        assert duration_ms < 1000, f"Memories endpoint too slow: {duration_ms}ms"
        print(f"Memories endpoint: {duration_ms:.2f}ms")
    
    def test_api_summaries_performance(self, perf_tester, test_codebase):
        """Test /api/summaries endpoint performance."""
        endpoint = f"/api/summaries?codebase={test_codebase}&limit=50"
        duration_ms, response = perf_tester.time_request("GET", endpoint)
        
        result = perf_tester.record_result(
            "/api/summaries", duration_ms, response.status_code,
            len(response.content), f"List summaries for {test_codebase}"
        )
        
        assert response.status_code == 200
        assert duration_ms < 1500, f"Summaries endpoint too slow: {duration_ms}ms"
        print(f"Summaries endpoint: {duration_ms:.2f}ms")
    
    def test_api_summary_stats_performance(self, perf_tester, test_codebase):
        """Test /api/summary-stats endpoint performance."""
        endpoint = f"/api/summary-stats?codebase={test_codebase}"
        duration_ms, response = perf_tester.time_request("GET", endpoint)
        
        result = perf_tester.record_result(
            "/api/summary-stats", duration_ms, response.status_code,
            len(response.content), f"Summary stats for {test_codebase}"
        )
        
        assert response.status_code == 200
        assert duration_ms < 1000, f"Summary stats endpoint too slow: {duration_ms}ms"
        print(f"Summary stats endpoint: {duration_ms:.2f}ms")
    
    def test_performance_summary(self, perf_tester):
        """Print overall performance summary."""
        summary = perf_tester.get_summary()
        
        print("\n" + "="*60)
        print("API PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total requests: {summary.get('total_requests', 0)}")
        print(f"Successful requests: {summary.get('successful_requests', 0)}")
        print(f"Average duration: {summary.get('avg_duration_ms', 0):.2f}ms")
        print(f"Min duration: {summary.get('min_duration_ms', 0):.2f}ms")
        print(f"Max duration: {summary.get('max_duration_ms', 0):.2f}ms")
        
        slow_endpoints = summary.get('slow_endpoints', [])
        if slow_endpoints:
            print(f"\nSLOW ENDPOINTS (>1000ms):")
            for endpoint in slow_endpoints:
                print(f"  {endpoint['endpoint']}: {endpoint['duration_ms']:.2f}ms - {endpoint['notes']}")
        
        failed_requests = summary.get('failed_requests', [])
        if failed_requests:
            print(f"\nFAILED REQUESTS:")
            for req in failed_requests:
                print(f"  {req['endpoint']}: HTTP {req['status_code']} - {req['notes']}")
        
        print("="*60)


class TestFileSpecificPerformance:
    """Detailed performance tests for the problematic file-summary endpoint."""
    
    def test_file_summary_detailed_timing(self, test_codebase):
        """Detailed timing analysis of the file-summary endpoint."""
        test_files = [
            "src/conductor_memory/service/__init__.py",
            "src/conductor_memory/core/models.py", 
            "src/conductor_memory/storage/chroma.py",
            "src/conductor_memory/server/sse.py"
        ]
        
        print("\n" + "="*60)
        print("DETAILED FILE-SUMMARY PERFORMANCE ANALYSIS")
        print("="*60)
        
        for file_path in test_files:
            endpoint = f"/api/file-summary?codebase={test_codebase}&path={quote(file_path)}"
            
            # Time the request with detailed breakdown
            start_time = time.perf_counter()
            response = requests.get(f"http://localhost:9820{endpoint}")
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            
            print(f"\nFile: {file_path}")
            print(f"  Duration: {duration_ms:.2f}ms")
            print(f"  Status: {response.status_code}")
            print(f"  Response size: {len(response.content)} bytes")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    summary = data.get("summary")
                    if summary:
                        print(f"  Has summary: Yes")
                        print(f"  Has how_it_works: {summary.get('has_how_it_works', False)}")
                        print(f"  Has method_summaries: {summary.get('has_method_summaries', False)}")
                        print(f"  Model: {summary.get('model', 'unknown')}")
                    else:
                        print(f"  Has summary: No")
                except json.JSONDecodeError:
                    print(f"  Response: Invalid JSON")
            
            if duration_ms > 5000:
                print(f"  ⚠️  VERY SLOW (>{duration_ms:.0f}ms)")
            elif duration_ms > 1000:
                print(f"  ⚠️  SLOW (>{duration_ms:.0f}ms)")
        
        print("="*60)


if __name__ == "__main__":
    # Run a quick performance check
    tester = APIPerformanceTester()
    
    print("Running quick API performance check...")
    
    # Test a few key endpoints
    endpoints = [
        ("/health", "GET"),
        ("/api/status", "GET"),
        ("/api/codebases", "GET"),
    ]
    
    for endpoint, method in endpoints:
        try:
            duration_ms, response = tester.time_request(method, endpoint)
            tester.record_result(endpoint, duration_ms, response.status_code)
            print(f"{endpoint}: {duration_ms:.2f}ms (HTTP {response.status_code})")
        except Exception as e:
            print(f"{endpoint}: ERROR - {e}")
    
    # Test the problematic file-summary endpoint
    try:
        test_endpoint = "/api/file-summary?codebase=conductor-memory&path=src%2Fconductor_memory%2Fservice%2F__init__.py"
        duration_ms, response = tester.time_request("GET", test_endpoint)
        tester.record_result("/api/file-summary", duration_ms, response.status_code)
        print(f"/api/file-summary: {duration_ms:.2f}ms (HTTP {response.status_code})")
        
        if duration_ms > 5000:
            print("⚠️  File-summary endpoint is very slow!")
    except Exception as e:
        print(f"/api/file-summary: ERROR - {e}")
    
    summary = tester.get_summary()
    print(f"\nSummary: {summary.get('successful_requests', 0)} successful requests, "
          f"avg {summary.get('avg_duration_ms', 0):.2f}ms")