"""
Integration tests for the FastAPI server
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.model_server import app


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'timestamp' in data
    
    def test_infer_benign_query(self, client):
        """Test inference with benign query"""
        payload = {
            "method": "GET",
            "url": "/products",
            "params": {"search": "laptop"},
            "raw_query": "laptop"
        }
        response = client.post("/infer", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data['score'] < 0.5
        assert data['action'] == 'allow'
    
    def test_infer_sql_injection(self, client):
        """Test inference with SQL injection"""
        payload = {
            "method": "GET",
            "url": "/login",
            "params": {"username": "admin' OR '1'='1"},
            "raw_query": "admin' OR '1'='1"
        }
        response = client.post("/infer", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data['score'] >= 0.5
        assert data['action'] in ['block', 'challenge']
    
    def test_infer_union_attack(self, client):
        """Test inference with UNION attack"""
        payload = {
            "method": "GET",
            "url": "/api/users",
            "params": {"id": "1' UNION SELECT NULL, username, password FROM users--"},
            "raw_query": "1' UNION SELECT NULL, username, password FROM users--"
        }
        response = client.post("/infer", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data['score'] >= 0.9
        assert data['action'] == 'block'
        assert 'UNION SELECT' in data['matched_rules']
    
    def test_infer_missing_query(self, client):
        """Test inference with missing query parameters"""
        payload = {
            "method": "GET",
            "url": "/test"
        }
        response = client.post("/infer", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data['score'] == 0.0
        assert data['action'] == 'allow'
    
    def test_stats_endpoint(self, client):
        """Test the statistics endpoint"""
        # Make a few requests first
        client.post("/infer", json={
            "method": "GET",
            "url": "/test",
            "raw_query": "laptop"
        })
        client.post("/infer", json={
            "method": "GET",
            "url": "/test",
            "raw_query": "1' OR '1'='1"
        })
        
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert 'total_requests' in data
        assert 'blocked' in data
        assert 'allowed' in data
        assert data['total_requests'] >= 2
    
    def test_batch_endpoint(self, client):
        """Test batch inference endpoint"""
        payload = {
            "requests": [
                {
                    "method": "GET",
                    "url": "/search",
                    "raw_query": "laptop"
                },
                {
                    "method": "GET",
                    "url": "/login",
                    "raw_query": "admin' OR '1'='1"
                },
                {
                    "method": "GET",
                    "url": "/users",
                    "raw_query": "1' UNION SELECT NULL--"
                }
            ]
        }
        response = client.post("/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data['results']) == 3
        assert data['results'][0]['action'] == 'allow'
        assert data['results'][1]['action'] in ['block', 'challenge']
        assert data['results'][2]['action'] == 'block'
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
