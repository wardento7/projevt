"""
Unit tests for the SQL Injection Detection Engine
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.model_server import DetectionEngine


@pytest.fixture
def engine():
    """Create a detection engine instance"""
    return DetectionEngine()


class TestDetectionEngine:
    """Test suite for DetectionEngine class"""
    
    def test_benign_queries(self, engine):
        """Test that benign queries are not flagged"""
        benign_queries = [
            "SELECT * FROM products WHERE category='laptop'",
            "SELECT id, name FROM users WHERE active=true",
            "INSERT INTO orders (user_id, total) VALUES (1, 99.99)",
            "UPDATE profile SET email='user@example.com' WHERE id=5",
            "laptop",
            "search query",
            "product name",
        ]
        
        for query in benign_queries:
            score, matched = engine.detect(query)
            assert score < 0.5, f"False positive for: {query}"
    
    def test_sql_injection_attacks(self, engine):
        """Test that SQL injection attacks are detected"""
        malicious_queries = [
            "1' OR '1'='1",
            "admin' --",
            "1' UNION SELECT NULL, NULL--",
            "' OR 1=1--",
            "1'; DROP TABLE users--",
            "' OR '1'='1' /*",
            "1' AND 1=0 UNION SELECT NULL, username, password FROM users--",
        ]
        
        for query in malicious_queries:
            score, matched = engine.detect(query)
            assert score >= 0.5, f"Missed attack: {query}"
    
    def test_union_attacks(self, engine):
        """Test UNION-based SQL injection detection"""
        union_attacks = [
            "1' UNION SELECT NULL--",
            "1 UNION ALL SELECT NULL, NULL, NULL--",
            "' UNION SELECT username, password FROM users--",
        ]
        
        for query in union_attacks:
            score, matched = engine.detect(query)
            assert score >= 0.9, f"High-risk UNION attack not properly scored: {query}"
            assert 'UNION SELECT' in matched
    
    def test_time_based_attacks(self, engine):
        """Test time-based blind SQL injection detection"""
        time_attacks = [
            "1' AND SLEEP(5)--",
            "1' WAITFOR DELAY '00:00:05'--",
            "1' AND BENCHMARK(5000000, MD5('test'))--",
        ]
        
        for query in time_attacks:
            score, matched = engine.detect(query)
            assert score >= 0.8, f"Time-based attack not detected: {query}"
    
    def test_comment_patterns(self, engine):
        """Test SQL comment pattern detection"""
        comment_attacks = [
            "admin'--",
            "1' /*comment*/ OR '1'='1",
            "user'#",
            "' OR 1=1-- comment",
        ]
        
        for query in comment_attacks:
            score, matched = engine.detect(query)
            assert score > 0.5, f"Comment-based attack not detected: {query}"
    
    def test_empty_query(self, engine):
        """Test handling of empty queries"""
        score, matched = engine.detect("")
        assert score == 0.0
    
    def test_information_schema_access(self, engine):
        """Test detection of information_schema queries"""
        info_queries = [
            "' UNION SELECT table_name FROM information_schema.tables--",
            "1' AND 1=0 UNION SELECT NULL, column_name FROM information_schema.columns--",
        ]
        
        for query in info_queries:
            score, matched = engine.detect(query)
            assert score >= 0.9, f"Information schema access not detected: {query}"
    
    def test_stacked_queries(self, engine):
        """Test detection of stacked queries"""
        stacked = [
            "1'; DROP TABLE users--",
            "admin'; UPDATE users SET role='admin'--",
            "1; DELETE FROM logs;--",
        ]
        
        for query in stacked:
            score, matched = engine.detect(query)
            assert score >= 0.8, f"Stacked query not detected: {query}"
    
    def test_threshold_scoring(self, engine):
        """Test that scoring respects thresholds"""
        # Low-risk query
        low_score, _ = engine.detect("SELECT * FROM users WHERE id=1")
        assert low_score < 0.3
        
        # Medium-risk query (might need investigation)
        medium_score, _ = engine.detect("1' OR '1'='1")
        assert 0.5 <= medium_score < 0.95
        
        # High-risk query
        high_score, _ = engine.detect("1' UNION SELECT NULL, username, password FROM users--")
        assert high_score >= 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
