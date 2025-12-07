#!/usr/bin/env python3
"""
Model Benchmarking Script for Summarization Performance

This script runs independently from the conductor-memory server to benchmark
different Ollama models on a sample of files from your codebase.

Usage:
    python scripts/benchmark_models.py --codebase /path/to/codebase --sample 20

Requirements:
    - Ollama running locally (ollama serve)
    - Models to test pulled (ollama pull qwen2.5-coder:0.5b, etc.)
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
import fnmatch


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single model."""
    model: str
    files_processed: int
    files_failed: int
    total_time_seconds: float
    avg_time_per_file: float
    min_time: float
    max_time: float
    avg_tokens_used: float
    sample_summaries: List[Dict[str, Any]]


# Skip patterns (same as conductor-memory defaults)
SKIP_PATTERNS = [
    "**/test/**", "**/tests/**", "**/*_test.*", "**/test_*.*",
    "**/__pycache__/**", "**/build/**", "**/dist/**",
    "**/node_modules/**", "**/vendor/**", "**/.gradle/**",
    "**/generated/**", "**/debug/**", "**/release/**"
]

# Code extensions to include
CODE_EXTENSIONS = {'.kt', '.java', '.py', '.go', '.swift', '.ts', '.js', '.tsx', '.jsx'}


def should_skip(file_path: str) -> bool:
    """Check if file matches skip patterns."""
    for pattern in SKIP_PATTERNS:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def collect_sample_files(codebase_path: Path, sample_size: int) -> List[Path]:
    """Collect a random sample of code files from the codebase."""
    all_files = []
    
    for f in codebase_path.rglob('*'):
        if not f.is_file():
            continue
        if f.suffix.lower() not in CODE_EXTENSIONS:
            continue
        
        rel_path = str(f.relative_to(codebase_path)).replace('\\', '/')
        if should_skip(rel_path):
            continue
        
        # Skip very large files (>1000 lines)
        try:
            line_count = len(f.read_text(encoding='utf-8', errors='ignore').split('\n'))
            if line_count > 1000:
                continue
        except:
            continue
        
        all_files.append(f)
    
    # Random sample
    if len(all_files) <= sample_size:
        return all_files
    return random.sample(all_files, sample_size)


async def call_ollama(
    session: aiohttp.ClientSession,
    model: str,
    file_path: str,
    content: str,
    base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """Call Ollama API to summarize a file."""
    
    system_prompt = """You are a code analysis expert. Analyze the provided code file and generate a structured summary in JSON format.
Respond with valid JSON only, no additional text."""

    user_prompt = f"""Analyze this code file and provide a structured summary.

File: {file_path}
```
{content[:4000]}
```

Respond with JSON only:
{{
  "purpose": "1-2 sentence description",
  "pattern": "architectural pattern (e.g., Repository, ViewModel, Controller)",
  "key_exports": ["main", "public", "APIs"],
  "dependencies": ["key", "dependencies"],
  "domain": "business domain"
}}"""

    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 384
        }
    }
    
    start_time = time.time()
    
    try:
        async with session.post(f"{base_url}/api/generate", json=payload) as response:
            if response.status != 200:
                return {"error": f"HTTP {response.status}", "time": 0}
            
            data = await response.json()
            elapsed = time.time() - start_time
            
            tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
            
            # Try to parse JSON response
            raw_response = data.get("response", "")
            parsed_json = None
            try:
                parsed_json = json.loads(raw_response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group())
                    except:
                        pass
            
            return {
                "success": True,
                "time": elapsed,
                "tokens_used": tokens_used,
                "response_raw": raw_response,
                "response_parsed": parsed_json
            }
    except Exception as e:
        return {"error": str(e), "time": time.time() - start_time}


async def warm_up_model(session: aiohttp.ClientSession, model: str, base_url: str = "http://localhost:11434") -> bool:
    """Warm up a model before benchmarking."""
    print(f"  Warming up {model}...")
    
    payload = {
        "model": model,
        "prompt": "Say 'ready'",
        "stream": False,
        "options": {"num_predict": 5}
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=120)
        async with session.post(f"{base_url}/api/generate", json=payload, timeout=timeout) as response:
            if response.status == 200:
                print(f"  {model} is ready")
                return True
            return False
    except Exception as e:
        print(f"  Failed to warm up {model}: {e}")
        return False


async def benchmark_model(
    model: str,
    files: List[Path],
    codebase_path: Path,
    base_url: str = "http://localhost:11434"
) -> BenchmarkResult:
    """Benchmark a single model on the sample files."""
    
    print(f"\nBenchmarking: {model}")
    print(f"  Files to process: {len(files)}")
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warm up the model first
        if not await warm_up_model(session, model, base_url):
            return BenchmarkResult(
                model=model,
                files_processed=0,
                files_failed=len(files),
                total_time_seconds=0,
                avg_time_per_file=0,
                min_time=0,
                max_time=0,
                avg_tokens_used=0,
                sample_summaries=[]
            )
        
        times = []
        tokens = []
        failures = 0
        sample_summaries = []
        
        for i, file_path in enumerate(files):
            rel_path = str(file_path.relative_to(codebase_path))
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                content_preview = '\n'.join(content.split('\n')[:50])  # First 50 lines for review
            except:
                failures += 1
                continue
            
            result = await call_ollama(session, model, rel_path, content, base_url)
            result["content_preview"] = content_preview  # Store for quality review
            
            if "error" in result:
                failures += 1
                print(f"  [{i+1}/{len(files)}] FAILED: {rel_path} - {result['error']}")
                sample_summaries.append({
                    "file": rel_path,
                    "time": result.get("time", 0),
                    "error": result.get("error"),
                    "response_raw": None,
                    "response_parsed": None
                })
            else:
                times.append(result["time"])
                tokens.append(result.get("tokens_used", 0))
                
                # Store ALL summaries for quality comparison
                sample_summaries.append({
                    "file": rel_path,
                    "time": result["time"],
                    "tokens_used": result.get("tokens_used", 0),
                    "content_preview": result.get("content_preview", ""),
                    "response_raw": result.get("response_raw", ""),
                    "response_parsed": result.get("response_parsed")
                })
                
                print(f"  [{i+1}/{len(files)}] {result['time']:.2f}s - {rel_path}")
            
            # Small delay between requests
            await asyncio.sleep(0.05)
    
    if not times:
        return BenchmarkResult(
            model=model,
            files_processed=0,
            files_failed=failures,
            total_time_seconds=0,
            avg_time_per_file=0,
            min_time=0,
            max_time=0,
            avg_tokens_used=0,
            sample_summaries=sample_summaries
        )
    
    return BenchmarkResult(
        model=model,
        files_processed=len(times),
        files_failed=failures,
        total_time_seconds=sum(times),
        avg_time_per_file=sum(times) / len(times),
        min_time=min(times),
        max_time=max(times),
        avg_tokens_used=sum(tokens) / len(tokens) if tokens else 0,
        sample_summaries=sample_summaries
    )


async def check_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Get list of available models from Ollama."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Sort by avg time
    results = sorted(results, key=lambda r: r.avg_time_per_file if r.avg_time_per_file > 0 else 999)
    
    print(f"\n{'Model':<25} {'Avg Time':<12} {'Min':<10} {'Max':<10} {'Success':<10} {'Tokens':<10}")
    print("-" * 80)
    
    for r in results:
        if r.files_processed == 0:
            print(f"{r.model:<25} {'FAILED':<12} {'-':<10} {'-':<10} {0}/{r.files_failed:<10}")
        else:
            success_rate = f"{r.files_processed}/{r.files_processed + r.files_failed}"
            print(f"{r.model:<25} {r.avg_time_per_file:.2f}s{'':<6} {r.min_time:.2f}s{'':<4} {r.max_time:.2f}s{'':<4} {success_rate:<10} {r.avg_tokens_used:.0f}")
    
    print("\n" + "-" * 80)
    
    # Recommendation
    best = results[0] if results and results[0].files_processed > 0 else None
    if best:
        current_time = 1.1  # Current baseline
        speedup = current_time / best.avg_time_per_file if best.avg_time_per_file > 0 else 1
        print(f"\nFASTEST: {best.model} at {best.avg_time_per_file:.2f}s/file")
        if speedup > 1:
            print(f"         {speedup:.1f}x faster than current (1.1s/file baseline)")
        
        # Extrapolate to full codebase
        estimated_files = 2000  # Approximate for truthsocial-android
        est_time_minutes = (best.avg_time_per_file * estimated_files) / 60
        print(f"         Estimated time for ~{estimated_files} files: {est_time_minutes:.0f} minutes")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models for code summarization")
    parser.add_argument("--codebase", type=str, required=True, help="Path to codebase to sample from")
    parser.add_argument("--sample", type=int, default=20, help="Number of files to sample (default: 20)")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:3b"],
                        help="Models to benchmark")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    codebase_path = Path(args.codebase)
    if not codebase_path.exists():
        print(f"Error: Codebase path does not exist: {codebase_path}")
        return
    
    print(f"Codebase: {codebase_path}")
    print(f"Sample size: {args.sample}")
    print(f"Models to test: {args.models}")
    
    # Check available models
    available = await check_available_models(args.ollama_url)
    if not available:
        print("\nError: Cannot connect to Ollama. Is it running?")
        print("Start with: ollama serve")
        return
    
    print(f"\nAvailable models: {available}")
    
    # Filter to available models
    models_to_test = [m for m in args.models if m in available]
    missing = [m for m in args.models if m not in available]
    
    if missing:
        print(f"\nModels not available (pull with 'ollama pull <model>'): {missing}")
    
    if not models_to_test:
        print("\nNo requested models are available. Pull them first:")
        for m in args.models:
            print(f"  ollama pull {m}")
        return
    
    # Collect sample files
    print(f"\nCollecting sample files...")
    files = collect_sample_files(codebase_path, args.sample)
    print(f"Selected {len(files)} files for benchmarking")
    
    if not files:
        print("No suitable files found in codebase")
        return
    
    # Benchmark each model
    results = []
    for model in models_to_test:
        result = await benchmark_model(model, files, codebase_path, args.ollama_url)
        results.append(result)
    
    # Print results
    print_results(results)
    
    # Save to file if requested
    if args.output:
        from datetime import datetime
        
        output_data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "codebase": str(codebase_path),
                "sample_size": len(files),
                "files_sampled": [str(f.relative_to(codebase_path)) for f in files]
            },
            "performance_summary": [
                {
                    "model": r.model,
                    "files_processed": r.files_processed,
                    "files_failed": r.files_failed,
                    "total_time_seconds": round(r.total_time_seconds, 2),
                    "avg_time_per_file": round(r.avg_time_per_file, 3),
                    "min_time": round(r.min_time, 3),
                    "max_time": round(r.max_time, 3),
                    "avg_tokens_used": round(r.avg_tokens_used, 0)
                }
                for r in results
            ],
            "quality_data": {
                r.model: r.sample_summaries
                for r in results
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
        print(f"  - Performance summary for quick comparison")
        print(f"  - Full quality_data with source previews + summaries for each model")
        print(f"  - Use this file for AI-assisted quality analysis")


if __name__ == "__main__":
    asyncio.run(main())
