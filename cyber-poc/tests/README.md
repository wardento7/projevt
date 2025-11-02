# 🧪 Test Suite Documentation

## Overview
Comprehensive test suite for the Cyber Security POC project with **17 automated tests** covering API endpoints and detection engine functionality.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── test_api.py                    # API endpoint tests (8 tests)
├── test_detection_engine.py       # Detection engine tests (9 tests)
└── README.md                      # This file
```

## Running Tests

### Run All Tests
```bash
# Using pytest
pytest tests/ -v

# Using make
make test-unit
```

### Run Specific Test File
```bash
# API tests only
pytest tests/test_api.py -v

# Detection engine tests only
pytest tests/test_detection_engine.py -v
```

### Run Specific Test
```bash
pytest tests/test_api.py::TestAPIEndpoints::test_health_endpoint -v
```

## Test Coverage

### API Tests (8 tests)
| Test | Description | Status |
|------|-------------|--------|
| `test_health_endpoint` | Verify health check returns correct status | ✅ |
| `test_infer_benign_query` | Test benign query detection | ✅ |
| `test_infer_sql_injection` | Test SQL injection detection | ✅ |
| `test_infer_union_attack` | Test UNION attack detection | ✅ |
| `test_infer_missing_query` | Test handling of missing query | ✅ |
| `test_stats_endpoint` | Test statistics endpoint | ✅ |
| `test_batch_endpoint` | Test batch processing | ✅ |
| `test_cors_headers` | Test CORS configuration | ✅ |

### Detection Engine Tests (9 tests)
| Test | Description | Status |
|------|-------------|--------|
| `test_benign_queries` | Verify benign queries pass | ✅ |
| `test_sql_injection_attacks` | Detect various SQL injections | ✅ |
| `test_union_attacks` | Detect UNION-based attacks | ✅ |
| `test_time_based_attacks` | Detect time-based blind SQLi | ✅ |
| `test_comment_patterns` | Detect SQL comment patterns | ✅ |
| `test_empty_query` | Handle empty queries | ✅ |
| `test_information_schema_access` | Detect info schema queries | ✅ |
| `test_stacked_queries` | Detect stacked query attacks | ✅ |
| `test_threshold_scoring` | Verify scoring thresholds | ✅ |

## Test Results

```
==================== 17 passed in 0.22s ====================

Success Rate: 100%
Total Tests: 17
Passed: 17
Failed: 0
```

## Adding New Tests

### API Test Example
```python
def test_new_endpoint(self, client):
    """Test description"""
    response = client.get("/new-endpoint")
    assert response.status_code == 200
    assert response.json()['key'] == 'expected_value'
```

### Detection Engine Test Example
```python
def test_new_pattern(self, engine):
    """Test description"""
    score, matched = engine.detect("malicious query")
    assert score >= 0.5
    assert 'pattern_name' in matched
```

## Continuous Integration

To integrate with CI/CD:

```yaml
# .github/workflows/test.yml
- name: Run Tests
  run: |
    source .venv/bin/activate
    pytest tests/ -v --cov=backend
```

## Test Configuration

Tests use:
- **pytest** framework
- **TestClient** from FastAPI for API testing
- **Fixtures** for reusable test components

## Debugging Failed Tests

```bash
# Verbose output with full traceback
pytest tests/ -vv --tb=long

# Stop at first failure
pytest tests/ -x

# Run only failed tests
pytest tests/ --lf
```

## Performance

- Average test execution: **0.22 seconds**
- All tests run in parallel where possible
- No external dependencies required

## Notes

- Tests use in-memory FastAPI TestClient (no server required)
- Detection engine tests are isolated and stateless
- All tests pass with Python 3.14
