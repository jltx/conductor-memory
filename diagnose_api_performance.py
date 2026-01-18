"""
API Performance Diagnostic Script
Run this while the server is up to identify bottlenecks.

Usage: uv run python diagnose_api_performance.py
"""
import sys
import io

# Fix Windows UTF-8 encoding for emoji
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import requests
from typing import Dict, Any, Tuple

BASE_URL = "http://127.0.0.1:9820"


def measure_endpoint(name: str, url: str, method: str = "GET", data: dict = None) -> Tuple[float, int, Any]:
    """Measure response time for an endpoint."""
    start = time.perf_counter()
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, response.status_code, response.json() if response.ok else None
    except requests.exceptions.Timeout:
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, -1, "TIMEOUT"
    except requests.exceptions.ConnectionError:
        return -1, -2, "CONNECTION_ERROR"
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, -3, str(e)


def check_server() -> bool:
    """Check if server is responding."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.ok
    except:
        return False


def main():
    print("=" * 70)
    print("API Performance Diagnostic")
    print("=" * 70)
    print()

    # Check server is up
    print("Checking server availability...")
    if not check_server():
        print("ERROR: Server not responding at", BASE_URL)
        print("Please start the server first:")
        print("  uv run python -m conductor_memory.server.sse")
        return 1

    print(f"Server is up at {BASE_URL}")
    print()

    # Define endpoints to test
    endpoints = [
        ("Health Check", "/health", "GET"),
        ("Status (Full)", "/api/status", "GET"),
        ("Status Counts (Fast)", "/api/status/counts", "GET"),
        ("Summarization", "/api/summarization", "GET"),
        ("ChromaDB Status", "/api/chroma-status", "GET"),
        ("Codebases List", "/api/codebases", "GET"),
    ]

    # Timing thresholds (ms)
    FAST = 100
    ACCEPTABLE = 500
    SLOW = 2000

    results = []

    print("Testing endpoints (5 iterations each):")
    print("-" * 70)

    for name, path, method in endpoints:
        url = f"{BASE_URL}{path}"
        times = []
        status = None

        for i in range(5):
            elapsed, status_code, data = measure_endpoint(name, url, method)
            if elapsed > 0:
                times.append(elapsed)
            status = status_code
            time.sleep(0.2)  # Small delay between requests

        if times:
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)

            # Determine status indicator
            if avg < FAST:
                indicator = "[OK]"
            elif avg < ACCEPTABLE:
                indicator = "[SLOW]"
            elif avg < SLOW:
                indicator = "[VERY SLOW]"
            else:
                indicator = "[CRITICAL]"

            print(f"{name:25} avg: {avg:7.0f}ms  (min: {min_t:5.0f}ms, max: {max_t:5.0f}ms) {indicator}")
            results.append((name, avg, min_t, max_t, status))
        else:
            print(f"{name:25} FAILED (status: {status})")
            results.append((name, -1, -1, -1, status))

    print()
    print("=" * 70)
    print("Analysis:")
    print("=" * 70)
    print()

    # Analyze results
    critical_issues = []
    slow_issues = []

    for name, avg, min_t, max_t, status in results:
        if avg < 0:
            critical_issues.append(f"- {name}: Failed to respond")
        elif avg > SLOW:
            critical_issues.append(f"- {name}: {avg:.0f}ms (expected <{ACCEPTABLE}ms)")
        elif avg > ACCEPTABLE:
            slow_issues.append(f"- {name}: {avg:.0f}ms (expected <{FAST}ms)")

    if critical_issues:
        print("CRITICAL PERFORMANCE ISSUES:")
        for issue in critical_issues:
            print(issue)
        print()

    if slow_issues:
        print("SLOW ENDPOINTS:")
        for issue in slow_issues:
            print(issue)
        print()

    # Check variance (indicates contention)
    high_variance = []
    for name, avg, min_t, max_t, status in results:
        if avg > 0 and max_t > avg * 2:
            high_variance.append(f"- {name}: max ({max_t:.0f}ms) is 2x+ avg ({avg:.0f}ms)")

    if high_variance:
        print("HIGH VARIANCE (possible contention):")
        for issue in high_variance:
            print(issue)
        print()

    if not critical_issues and not slow_issues:
        print("All endpoints performing within acceptable thresholds.")
        print()

    # Additional diagnostics
    print("=" * 70)
    print("Additional Checks:")
    print("=" * 70)
    print()

    # Check ChromaDB status
    _, _, chroma_data = measure_endpoint("ChromaDB", f"{BASE_URL}/api/chroma-status")
    if chroma_data and isinstance(chroma_data, dict):
        print(f"ChromaDB Mode: {chroma_data.get('mode', 'unknown')}")
        if chroma_data.get('mode') == 'http':
            print(f"  Host: {chroma_data.get('host')}:{chroma_data.get('port')}")
            print(f"  Connected: {chroma_data.get('connected', False)}")
            print(f"  Server Healthy: {chroma_data.get('server_healthy', False)}")
        print()

    # Check status for more details
    _, _, status_data = measure_endpoint("Status", f"{BASE_URL}/api/status")
    if status_data and isinstance(status_data, dict):
        codebases = status_data.get('codebases', {})
        print(f"Codebases: {len(codebases)}")
        total_files = 0
        for name, info in codebases.items():
            count = info.get('indexed_files_count', 0)
            total_files += count
            print(f"  {name}: {count} files")
        print(f"  Total: {total_files} files")
        print()

    # Check summarization status
    _, _, summ_data = measure_endpoint("Summarization", f"{BASE_URL}/api/summarization")
    if summ_data and isinstance(summ_data, dict):
        print(f"Summarization:")
        print(f"  Enabled: {summ_data.get('enabled', False)}")
        print(f"  Running: {summ_data.get('is_running', False)}")
        print(f"  Queued: {summ_data.get('files_queued', 0)}")
        print(f"  Current: {summ_data.get('current_file', 'None')}")
        print()

    print("=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print()

    # Generate recommendations based on findings
    has_recommendations = False

    if any(avg > SLOW for _, avg, _, _, _ in results if avg > 0):
        has_recommendations = True
        print("1. CHECK POSTGRESQL CONNECTION:")
        print("   Your config has postgres_url configured.")
        print("   If the host is unavailable, lazy init will timeout (~10s)")
        print("   Verify the PostgreSQL server is reachable.")
        print()

    if high_variance:
        has_recommendations = True
        print("2. CHECK FOR CONTENTION:")
        print("   High variance suggests database lock contention")
        print("   If summarization is running, try pausing it")
        print()

    if chroma_data and chroma_data.get('mode') == 'http' and not chroma_data.get('server_healthy'):
        has_recommendations = True
        print("3. CHROMADB HTTP SERVER ISSUE:")
        print("   ChromaDB HTTP server may not be healthy")
        print("   Check if chroma server is running on port 8100")
        print()

    if not has_recommendations:
        print("No specific issues detected in this run.")
        print("If you're still experiencing slow responses:")
        print("  - Check server logs for errors")
        print("  - Monitor network latency to PostgreSQL host")
        print("  - Try running with summarization disabled")

    return 0


if __name__ == "__main__":
    sys.exit(main())
