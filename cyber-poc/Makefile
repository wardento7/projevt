.PHONY: setup recon gen-data gen-payloads run-server test test-unit clean help

# Configuration variables
TARGET ?= http://localhost:8000
NUM_BENIGN ?= 700000
NUM_MALICIOUS ?= 300000
CHUNK_SIZE ?= 10000
AUGMENT ?= 5
BOT_IP_RATIO ?= 0.1
START_DATE ?= 2025-01-01
END_DATE ?= 2025-10-29
WHITE_LIST ?= localhost 127.0.0.1

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Cyber Security Intelligent Threat Mitigation$(NC)"
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Configuration variables:"
	@echo "  TARGET=$(TARGET)"
	@echo "  NUM_BENIGN=$(NUM_BENIGN)"
	@echo "  NUM_MALICIOUS=$(NUM_MALICIOUS)"
	@echo "  AUGMENT=$(AUGMENT)"

setup: ## Setup environment and install dependencies
	@echo "$(GREEN)Setting up environment...$(NC)"
	@python3 -m venv .venv || echo "$(YELLOW)venv already exists$(NC)"
	@echo "$(GREEN)Activating virtual environment...$(NC)"
	@( \
		. .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r backend/requirements-backend.txt && \
		echo "$(GREEN)✓ Backend dependencies installed$(NC)" \
	)
	@mkdir -p recon/output
	@mkdir -p data
	@mkdir -p logs
	@mkdir -p deliverables
	@echo "$(GREEN)✓ Directory structure created$(NC)"
	@echo "$(GREEN)✓ Setup completed successfully!$(NC)"
	@echo ""
	@echo "Optional tools (install manually if needed):"
	@echo "  - nmap: brew install nmap (macOS) or apt install nmap (Linux)"
	@echo "  - OWASP ZAP: https://www.zaproxy.org/download/"
	@echo "  - sqlmap: brew install sqlmap (macOS) or apt install sqlmap (Linux)"

gen-payloads: ## Generate payload database
	@echo "$(GREEN)Generating payloads...$(NC)"
	@. .venv/bin/activate && python3 recon/generate_payloads.py \
		--min-base 200 \
		--variants-per-base 5 \
		--output recon/payloads.json
	@echo "$(GREEN)✓ Payloads generated$(NC)"

gen-data: gen-payloads ## Generate synthetic dataset
	@echo "$(GREEN)Generating synthetic dataset...$(NC)"
	@echo "This may take several minutes for large datasets..."
	@. .venv/bin/activate && python3 data/generate_synthetic_dataset.py \
		--num-benign $(NUM_BENIGN) \
		--num-malicious $(NUM_MALICIOUS) \
		--chunk-size $(CHUNK_SIZE) \
		--augment-multiplier $(AUGMENT) \
		--bot-ip-ratio $(BOT_IP_RATIO) \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--out-csv data/dataset.csv \
		--out-jl data/dataset.jl
	@echo "$(GREEN)✓ Dataset generated successfully!$(NC)"
	@echo "Files created:"
	@ls -lh data/dataset.csv data/dataset.jl data/generation_report.json 2>/dev/null || true

recon: ## Run reconnaissance tools (nmap, zap, sqlmap)
	@echo "$(GREEN)Running reconnaissance...$(NC)"
	@echo "⚠️  Target: $(TARGET)"
	@echo "⚠️  Whitelist: $(WHITE_LIST)"
	@echo ""
	@echo "$(YELLOW)Running nmap...$(NC)"
	@. .venv/bin/activate && python3 recon/run_nmap.py \
		--target $(TARGET) \
		--white-list $(WHITE_LIST) || echo "$(YELLOW)nmap scan completed with warnings$(NC)"
	@echo ""
	@echo "$(YELLOW)Running OWASP ZAP...$(NC)"
	@chmod +x recon/run_zap.sh
	@bash recon/run_zap.sh \
		--target $(TARGET) \
		--white-list "$(WHITE_LIST)" || echo "$(YELLOW)ZAP scan completed with warnings$(NC)"
	@echo ""
	@echo "$(YELLOW)Running sqlmap...$(NC)"
	@chmod +x recon/run_sqlmap.sh
	@bash recon/run_sqlmap.sh \
		--target $(TARGET) \
		--white-list "$(WHITE_LIST)" || echo "$(YELLOW)sqlmap scan completed with warnings$(NC)"
	@echo ""
	@echo "$(GREEN)✓ Reconnaissance completed!$(NC)"
	@echo "Results saved in: recon/output/"

run-server: ## Run the FastAPI server
	@echo "$(GREEN)Starting FastAPI server...$(NC)"
	@echo "Server will be available at: http://localhost:8000"
	@echo "API docs: http://localhost:8000/docs"
	@echo ""
	@. .venv/bin/activate && uvicorn backend.model_server:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

test: ## Run inference tests
	@echo "$(GREEN)Running inference tests...$(NC)"
	@. .venv/bin/activate && python3 -c "\

test-unit: ## Run unit and integration tests with pytest
	@echo "$(GREEN)Running unit tests with pytest...$(NC)"
	@. .venv/bin/activate && pytest tests/ -v --tb=short
	@echo "$(GREEN)✓ All tests passed!$(NC)"

test-api: ## Run API integration tests
	@echo "$(GREEN)Running API integration tests...$(NC)"
	@. .venv/bin/activate && python3 -c "\
import requests; \
import json; \
import time; \
from datetime import datetime; \
\
print('Testing API endpoints...\\n'); \
\
# Health check \
print('1. Health check:'); \
resp = requests.get('http://localhost:8000/health'); \
print(f'   Status: {resp.status_code}'); \
print(f'   Response: {resp.json()}\\n'); \
\
# Test benign request \
print('2. Testing benign request:'); \
benign = { \
    'method': 'GET', \
    'url': '/search', \
    'params': {'q': 'laptop'}, \
    'body': None, \
    'raw_query': 'GET /search?q=laptop HTTP/1.1', \
    'source_ip': '192.168.1.100' \
}; \
resp = requests.post('http://localhost:8000/infer', json=benign); \
result = resp.json(); \
print(f'   Score: {result[\"score\"]}'); \
print(f'   Action: {result[\"action\"]}'); \
print(f'   Matched rules: {result[\"matched_rules\"]}\\n'); \
\
# Test malicious requests \
malicious_tests = [ \
    { \
        'name': 'Boolean-based blind', \
        'request': { \
            'method': 'GET', \
            'url': '/product', \
            'params': {'id': \"1' OR '1'='1\"}, \
            'body': None, \
            'raw_query': \"GET /product?id=1' OR '1'='1 HTTP/1.1\", \
            'source_ip': '10.0.0.1' \
        } \
    }, \
    { \
        'name': 'UNION SELECT', \
        'request': { \
            'method': 'GET', \
            'url': '/item', \
            'params': {'id': \"1' UNION SELECT NULL,NULL,NULL--\"}, \
            'body': None, \
            'raw_query': \"GET /item?id=1' UNION SELECT NULL,NULL,NULL-- HTTP/1.1\", \
            'source_ip': '10.0.0.2' \
        } \
    }, \
    { \
        'name': 'Time-based blind', \
        'request': { \
            'method': 'GET', \
            'url': '/search', \
            'params': {'q': \"test' AND SLEEP(5)--\"}, \
            'body': None, \
            'raw_query': \"GET /search?q=test' AND SLEEP(5)-- HTTP/1.1\", \
            'source_ip': '10.0.0.3' \
        } \
    }, \
    { \
        'name': 'Comment injection', \
        'request': { \
            'method': 'POST', \
            'url': '/comment', \
            'params': {}, \
            'body': 'username=admin&comment=test--', \
            'raw_query': 'POST /comment HTTP/1.1\\nContent-Type: application/x-www-form-urlencoded\\n\\nusername=admin&comment=test--', \
            'source_ip': '10.0.0.4' \
        } \
    }, \
]; \
\
with open('deliverables/test_results.jl', 'w') as f: \
    for i, test in enumerate(malicious_tests, 3): \
        print(f'{i}. Testing {test[\"name\"]}:'); \
        resp = requests.post('http://localhost:8000/infer', json=test['request']); \
        result = resp.json(); \
        print(f'   Score: {result[\"score\"]}'); \
        print(f'   Action: {result[\"action\"]}'); \
        print(f'   Matched rules: {result[\"matched_rules\"]}\\n'); \
        \
        test_entry = { \
            'timestamp': datetime.utcnow().isoformat(), \
            'test_name': test['name'], \
            'result': result \
        }; \
        f.write(json.dumps(test_entry) + '\\n'); \
\
print('✓ All tests completed!'); \
print('Results saved to: deliverables/test_results.jl'); \
\
# Get stats \
print('\\nServer statistics:'); \
resp = requests.get('http://localhost:8000/stats'); \
stats = resp.json(); \
for key, value in stats.items(): \
    print(f'  {key}: {value}'); \
" || echo "$(RED)✗ Tests failed. Make sure server is running: make run-server$(NC)"

clean: ## Clean generated files and logs
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	@rm -rf logs/*.jl
	@rm -rf recon/output/*
	@rm -rf data/*.tmp
	@rm -rf data/__pycache__
	@rm -rf recon/__pycache__
	@rm -rf backend/__pycache__
	@rm -rf .pytest_cache
	@echo "$(GREEN)✓ Cleaned successfully!$(NC)"

clean-all: clean ## Clean everything including dataset and venv
	@echo "$(RED)⚠️  This will delete dataset and virtual environment!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/dataset.csv data/dataset.jl data/dataset.parquet; \
		rm -rf .venv; \
		echo "$(GREEN)✓ Everything cleaned!$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled.$(NC)"; \
	fi

summary: ## Show project summary
	@echo "$(GREEN)Project Summary$(NC)"
	@echo "==============="
	@echo ""
	@if [ -f deliverables/summary.txt ]; then \
		cat deliverables/summary.txt; \
	else \
		echo "No summary file found. Run some tasks first."; \
	fi

inspect-samples: ## Show malicious samples for inspection
	@echo "$(GREEN)Malicious Samples for Inspection$(NC)"
	@if [ -f data/inspection_samples.json ]; then \
		. .venv/bin/activate && python3 -c "import json; samples = json.load(open('data/inspection_samples.json')); print(f'Total samples: {len(samples)}\\n'); [print(f'{i+1}. {s[\"attack_type\"]} via {s[\"insertion_point\"]}:\\n   Payload: {s[\"raw_query\"][:100]}...\\n') for i, s in enumerate(samples[:10])]"; \
	else \
		echo "No inspection samples found. Run: make gen-data"; \
	fi

# Default target
.DEFAULT_GOAL := help
