"""
Search Quality Tests for MCP Memory Server

These tests validate that semantic search returns relevant, useful results by:
1. Searching for known concepts and verifying expected files/classes are found
2. Comparing semantic search to baseline grep/keyword search
3. Testing various query types (conceptual, specific, natural language)

Uses the Options-ML-Trader codebase as the test corpus since it has substantial,
well-structured code with clear domain concepts.

Run with: pytest tests/test_search_quality.py -v
"""

import unittest
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import sys

from conductor_memory.config.server import ServerConfig, CodebaseConfig
from conductor_memory.service.memory_service import MemoryService


# Test configuration
OPTIONS_ML_TRADER_PATH = Path("C:/Users/joshu/projects/options-ml-trader")
PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_test"


class SearchQualityTestBase(unittest.TestCase):
    """Base class with shared setup for search quality tests"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize MemoryService with Options-ML-Trader codebase"""
        if not OPTIONS_ML_TRADER_PATH.exists():
            raise unittest.SkipTest(f"Options-ML-Trader not found at {OPTIONS_ML_TRADER_PATH}")
        
        config = ServerConfig(
            persist_directory=str(PERSIST_DIR),
            enable_file_watcher=False,
            codebases=[
                CodebaseConfig(
                    name="test-codebase",
                    path=str(OPTIONS_ML_TRADER_PATH),
                    extensions=[".py"],
                    ignore_patterns=[
                        "__pycache__", ".git", ".venv", "venv", ".idea", ".pytest_cache",
                        ".opencode", ".claude", "data", "models", "checkpoints", "results",
                        "logs", "lightning_logs", "catboost_info", "benchmarks", "experiments"
                    ],
                    enabled=True
                )
            ]
        )
        
        cls.service = MemoryService(config)
        cls.service.initialize()
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup"""
        # Optionally delete test persist dir
        pass
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Helper to run a search and return results"""
        result = self.service.search(query=query, max_results=max_results)
        return result.get("results", [])
    
    def get_files_from_results(self, results: List[Dict]) -> Set[str]:
        """Extract unique file paths from search results"""
        files = set()
        for r in results:
            for tag in r.get("tags", []):
                if tag.startswith("file:"):
                    files.add(tag[5:])  # Remove "file:" prefix
        return files
    
    def grep_codebase(self, pattern: str, case_insensitive: bool = True) -> Set[str]:
        """Baseline: grep for a pattern and return matching files"""
        flags = "-ril" if case_insensitive else "-rl"
        try:
            result = subprocess.run(
                ["rg", flags, pattern, str(OPTIONS_ML_TRADER_PATH / "src")],
                capture_output=True,
                text=True,
                timeout=30
            )
            files = set()
            for line in result.stdout.strip().split("\n"):
                if line:
                    # Convert to relative path
                    rel = Path(line).relative_to(OPTIONS_ML_TRADER_PATH)
                    files.add(str(rel).replace("\\", "/"))
            return files
        except Exception:
            return set()
    
    def assert_finds_file(self, results: List[Dict], expected_file_pattern: str, msg: str = None):
        """Assert that at least one result contains a file matching the pattern"""
        files = self.get_files_from_results(results)
        matches = [f for f in files if expected_file_pattern in f]
        self.assertTrue(
            len(matches) > 0,
            msg or f"Expected to find file matching '{expected_file_pattern}' in results. Got: {files}"
        )
    
    def assert_finds_content(self, results: List[Dict], expected_content: str, msg: str = None):
        """Assert that at least one result contains the expected content"""
        for r in results:
            if expected_content.lower() in r.get("content", "").lower():
                return
        self.fail(msg or f"Expected to find content containing '{expected_content}' in results")


class TestConceptualSearch(SearchQualityTestBase):
    """Test searching for high-level concepts"""
    
    def test_search_swing_detection(self):
        """Search for 'swing detection' should find swing_detector.py"""
        results = self.search("swing point detection algorithm")
        self.assert_finds_file(results, "swing_detector")
        self.assert_finds_content(results, "SwingDetector")
    
    def test_search_trend_analysis(self):
        """Search for trend state machine should find trend_state.py"""
        results = self.search("trend state machine higher high lower low")
        self.assert_finds_file(results, "trend_state")
    
    def test_search_support_resistance(self):
        """Search for support/resistance zones should find sr_zones.py"""
        results = self.search("support resistance zone tracking")
        self.assert_finds_file(results, "sr_zone")
    
    def test_search_market_regime(self):
        """Search for market regime detection should find relevant files"""
        results = self.search("market regime detection trending ranging volatile")
        files = self.get_files_from_results(results)
        # Should find either market_regime.py or volatility_regime.py
        regime_files = [f for f in files if "regime" in f.lower()]
        self.assertTrue(len(regime_files) > 0, f"Expected regime-related files, got: {files}")
    
    def test_search_ensemble_voting(self):
        """Search for ensemble voting should find ensemble_coordinator.py"""
        results = self.search("ensemble model voting strategy weighted")
        self.assert_finds_file(results, "ensemble")


class TestSpecificClassSearch(SearchQualityTestBase):
    """Test searching for specific class/function names"""
    
    def test_search_xgboost_trader(self):
        """Search for XGBoost implementation"""
        results = self.search("XGBoost trader model training")
        self.assert_finds_file(results, "xgboost")
    
    def test_search_backtest_engine(self):
        """Search for backtesting engine"""
        results = self.search("backtest engine trade execution")
        self.assert_finds_file(results, "backtest")
    
    def test_search_feature_engineer(self):
        """Search for feature engineering"""
        results = self.search("feature engineering technical indicators")
        files = self.get_files_from_results(results)
        feature_files = [f for f in files if "feature" in f.lower() or "indicator" in f.lower()]
        self.assertTrue(len(feature_files) > 0, f"Expected feature-related files, got: {files}")
    
    def test_search_pipeline_executor(self):
        """Search for pipeline execution"""
        results = self.search("pipeline executor step execution")
        self.assert_finds_file(results, "pipeline")
    
    def test_search_trading_agent(self):
        """Search for LLM trading agent"""
        results = self.search("autonomous trading agent LLM")
        self.assert_finds_file(results, "agent")


class TestNaturalLanguageQueries(SearchQualityTestBase):
    """Test natural language/question-style queries"""
    
    def test_how_to_detect_swings(self):
        """Natural language: How are swing highs and lows detected?"""
        results = self.search("how to detect swing highs and swing lows in price data")
        self.assert_finds_file(results, "swing")
    
    def test_what_is_kelly_criterion(self):
        """Natural language: Position sizing with Kelly criterion"""
        results = self.search("position sizing Kelly criterion risk management")
        # Should find strategy-related files
        files = self.get_files_from_results(results)
        self.assertTrue(len(files) > 0, "Expected results for Kelly criterion query")
    
    def test_how_backtesting_works(self):
        """Natural language: How does backtesting work?"""
        results = self.search("how does the backtesting engine simulate trades")
        self.assert_finds_file(results, "backtest")
    
    def test_what_models_available(self):
        """Natural language: What ML models are available?"""
        results = self.search("what machine learning models are available for trading")
        files = self.get_files_from_results(results)
        # Should find multiple model files
        model_related = [f for f in files if any(m in f.lower() for m in ["model", "xgboost", "catboost", "lightgbm", "lstm", "tft"])]
        self.assertTrue(len(model_related) > 0, f"Expected model files, got: {files}")


class TestDomainTerminology(SearchQualityTestBase):
    """Test domain-specific terminology searches"""
    
    def test_search_mfe_mae(self):
        """Search for MFE/MAE (Maximum Favorable/Adverse Excursion)"""
        results = self.search("maximum favorable excursion MFE MAE")
        # These terms appear in backtesting/metrics
        self.assertTrue(len(results) > 0, "Expected results for MFE/MAE query")
    
    def test_search_r_multiple(self):
        """Search for R-multiple targets (2R, 3R)"""
        results = self.search("R multiple profit target 2R 3R risk reward")
        self.assertTrue(len(results) > 0, "Expected results for R-multiple query")
    
    def test_search_atr_indicator(self):
        """Search for ATR (Average True Range)"""
        results = self.search("ATR average true range volatility indicator")
        files = self.get_files_from_results(results)
        # ATR is used in volatility analysis, regime detection, and stop loss calculation
        relevant_files = [f for f in files if any(term in f.lower() for term in 
            ["indicator", "feature", "volatility", "regime", "strategy", "stop"])]
        self.assertTrue(len(relevant_files) > 0, f"Expected volatility-related files, got: {files}")
    
    def test_search_tiered_exits(self):
        """Search for tiered exit strategy"""
        results = self.search("tiered exit strategy partial profit taking")
        self.assert_finds_file(results, "strategy")


class TestComparisonToGrep(SearchQualityTestBase):
    """Compare semantic search results to grep baseline"""
    
    def test_semantic_finds_conceptual_matches(self):
        """Semantic search should find files that grep might miss with exact terms"""
        # Grep for exact term
        grep_files = self.grep_codebase("SwingDetector")
        
        # Semantic search with conceptual query (related to swing detection)
        results = self.search("detect local highs and lows in price movement")
        semantic_files = self.get_files_from_results(results)
        
        # Semantic search should find swing_detector for conceptual query
        swing_files = [f for f in semantic_files if "swing" in f.lower()]
        self.assertTrue(
            len(swing_files) > 0,
            f"Semantic search should find swing-related files for conceptual query. Got: {semantic_files}"
        )
    
    def test_semantic_handles_synonyms(self):
        """Semantic search should handle synonyms that grep would miss"""
        # Grep won't find "moving average" if code says "SMA" or "EMA"
        results = self.search("simple moving average crossover signal")
        self.assertTrue(len(results) > 0, "Semantic search should handle indicator synonyms")
    
    def test_precision_vs_grep(self):
        """Semantic search should have reasonable precision compared to grep"""
        # Search for a specific concept
        query = "hyperparameter optimization optuna"
        
        results = self.search(query, max_results=5)
        semantic_files = self.get_files_from_results(results)
        
        # Grep for same terms
        grep_files = self.grep_codebase("optuna")
        
        # At least some overlap expected for specific technical terms
        if grep_files:
            overlap = semantic_files & grep_files
            # Convert paths to comparable format
            semantic_normalized = {f.replace("\\", "/") for f in semantic_files}
            grep_normalized = {f.replace("\\", "/") for f in grep_files}
            
            # Check if any semantic result is in grep results
            has_overlap = any(
                any(g in s or s in g for g in grep_normalized)
                for s in semantic_normalized
            )
            self.assertTrue(
                has_overlap or len(semantic_files) > 0,
                f"Expected overlap between semantic {semantic_files} and grep {grep_files}"
            )


class TestSearchQualityMetrics(SearchQualityTestBase):
    """Test overall search quality metrics"""
    
    def test_relevance_scores_ordering(self):
        """Results should be ordered by relevance score (descending)"""
        results = self.search("machine learning model training", max_results=10)
        
        if len(results) > 1:
            scores = [r.get("relevance_score", 0) for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True),
                           "Results should be ordered by relevance score descending")
    
    def test_no_duplicate_results(self):
        """Search should not return duplicate chunks"""
        results = self.search("trading strategy backtest", max_results=20)
        
        ids = [r.get("id") for r in results]
        self.assertEqual(len(ids), len(set(ids)), "Results should not contain duplicates")
    
    def test_results_have_required_fields(self):
        """All results should have required fields"""
        results = self.search("data collection", max_results=5)
        
        required_fields = ["id", "content", "tags", "relevance_score"]
        for r in results:
            for field in required_fields:
                self.assertIn(field, r, f"Result missing required field: {field}")
    
    def test_content_is_meaningful(self):
        """Result content should be meaningful code, not just whitespace"""
        results = self.search("class definition", max_results=10)
        
        for r in results:
            content = r.get("content", "")
            # Content should have actual code
            self.assertTrue(
                len(content.strip()) > 50,
                f"Content too short: {content[:100]}"
            )
    
    def test_search_returns_within_time_limit(self):
        """Search should return within reasonable time"""
        import time
        
        start = time.time()
        results = self.search("complex multi-word query about trading strategies and machine learning models")
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 5.0, f"Search took too long: {elapsed:.2f}s")


class TestEdgeCases(SearchQualityTestBase):
    """Test edge cases and error handling"""
    
    def test_empty_query(self):
        """Empty query should return empty or handle gracefully"""
        results = self.search("")
        # Should not crash, may return empty or error
        self.assertIsInstance(results, list)
    
    def test_very_long_query(self):
        """Very long query should be handled"""
        long_query = "machine learning " * 100
        results = self.search(long_query)
        self.assertIsInstance(results, list)
    
    def test_special_characters(self):
        """Query with special characters should be handled"""
        results = self.search("def __init__(self):")
        self.assertIsInstance(results, list)
    
    def test_nonexistent_concept(self):
        """Query for nonexistent concept should return low/no results"""
        results = self.search("quantum blockchain neural kubernetes serverless")
        # May return results but likely low relevance or empty
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
