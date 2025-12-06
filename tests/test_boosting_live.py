#!/usr/bin/env python3
"""
Live test script for Phase 1 boosting on real codebases.
Run this while the memory server is running.
"""

import requests
import json

BASE_URL = "http://localhost:9800"

def search(query: str, codebase: str = None, domain_boosts: dict = None, max_results: int = 5):
    """Execute a search against the memory server"""
    payload = {
        "query": query,
        "max_results": max_results
    }
    if codebase:
        payload["codebase"] = codebase
    if domain_boosts:
        payload["domain_boosts"] = domain_boosts
    
    response = requests.post(f"{BASE_URL}/search", json=payload)
    return response.json()

def print_results(results: dict, title: str):
    """Pretty print search results"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Found: {results.get('total_found', 0)} results")
    print(f"Search mode: {results.get('search_mode_used', 'unknown')}")
    print(f"Query time: {results.get('query_time_ms', 0):.1f}ms")
    print()
    
    for i, result in enumerate(results.get("results", []), 1):
        # Extract domain from tags
        domain_tags = [t for t in result.get("tags", []) if t.startswith("domain:")]
        domain = domain_tags[0] if domain_tags else "no-domain"
        
        # Extract file from tags
        file_tags = [t for t in result.get("tags", []) if t.startswith("file:")]
        file = file_tags[0].replace("file:", "") if file_tags else "unknown"
        
        score = result.get("relevance_score", 0)
        content_preview = result.get("content", "")[:80].replace("\n", " ")
        
        print(f"{i}. [{score:.3f}] {domain}")
        print(f"   File: {file}")
        print(f"   Preview: {content_preview}...")
        print()

def test_basic_search():
    """Test basic search without boosting"""
    print("\n" + "="*60)
    print("TEST 1: Basic search (no boosting)")
    print("="*60)
    
    query = "authentication"
    results = search(query, max_results=5)
    print_results(results, f"Query: '{query}' (default boosting)")

def test_boost_classes():
    """Test boosting classes higher"""
    print("\n" + "="*60)
    print("TEST 2: Boost classes, penalize tests")
    print("="*60)
    
    query = "authentication"
    
    # Without custom boost
    results_normal = search(query, max_results=5)
    print_results(results_normal, f"Query: '{query}' - Normal")
    
    # With class boost
    results_boosted = search(query, max_results=5, domain_boosts={"class": 2.0, "test": 0.3})
    print_results(results_boosted, f"Query: '{query}' - Boosted (class: 2.0, test: 0.3)")

def test_boost_functions():
    """Test boosting functions for implementation queries"""
    print("\n" + "="*60)
    print("TEST 3: Boost functions for implementation details")
    print("="*60)
    
    query = "error handling"
    
    # Boost functions and private methods (implementation details)
    results = search(query, max_results=5, domain_boosts={"function": 1.5, "private": 1.3, "test": 0.5})
    print_results(results, f"Query: '{query}' - Boosted (function: 1.5, private: 1.3)")

def test_penalize_tests():
    """Test penalizing tests heavily"""
    print("\n" + "="*60)
    print("TEST 4: Heavily penalize tests")
    print("="*60)
    
    query = "database connection"
    
    # Normal search
    results_normal = search(query, max_results=5)
    print_results(results_normal, f"Query: '{query}' - Normal")
    
    # Penalize tests
    results_no_tests = search(query, max_results=5, domain_boosts={"test": 0.1})
    print_results(results_no_tests, f"Query: '{query}' - Tests penalized (test: 0.1)")

def test_specific_codebase():
    """Test on a specific codebase"""
    print("\n" + "="*60)
    print("TEST 5: Search specific codebase with boosting")
    print("="*60)
    
    query = "model training"
    codebase = "Options-ML-Trader"
    
    # Boost classes (likely model definitions)
    results = search(query, codebase=codebase, max_results=5, domain_boosts={"class": 1.8, "function": 1.2})
    print_results(results, f"Query: '{query}' in {codebase} - Boosted classes")

def compare_domain_distribution():
    """Compare domain distribution with and without boosting"""
    print("\n" + "="*60)
    print("TEST 6: Domain distribution comparison")
    print("="*60)
    
    query = "user interface"
    
    # Get more results to see distribution
    results_normal = search(query, max_results=10)
    results_boosted = search(query, max_results=10, domain_boosts={"class": 2.0, "function": 1.5, "test": 0.2})
    
    def count_domains(results):
        domains = {}
        for r in results.get("results", []):
            for tag in r.get("tags", []):
                if tag.startswith("domain:"):
                    domain = tag.replace("domain:", "")
                    domains[domain] = domains.get(domain, 0) + 1
        return domains
    
    normal_domains = count_domains(results_normal)
    boosted_domains = count_domains(results_boosted)
    
    print(f"\nQuery: '{query}'")
    print(f"\nDomain distribution WITHOUT boosting:")
    for domain, count in sorted(normal_domains.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")
    
    print(f"\nDomain distribution WITH boosting (class: 2.0, function: 1.5, test: 0.2):")
    for domain, count in sorted(boosted_domains.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")

def main():
    print("="*60)
    print("PHASE 1 BOOSTING LIVE TEST")
    print("="*60)
    
    # Check server is running
    try:
        status = requests.get(f"{BASE_URL}/status").json()
        print(f"\nServer status: {status.get('status', 'unknown')}")
        print(f"Codebases indexed: {list(status.get('codebases', {}).keys())}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the memory server is running on port 9800")
        return
    
    # Run tests
    test_basic_search()
    test_boost_classes()
    test_boost_functions()
    test_penalize_tests()
    test_specific_codebase()
    compare_domain_distribution()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nObserve how domain boosting affects:")
    print("1. Result ordering (higher boosted domains should appear first)")
    print("2. Relevance scores (boosted items should have higher scores)")
    print("3. Domain distribution (fewer tests when penalized)")

if __name__ == "__main__":
    main()
