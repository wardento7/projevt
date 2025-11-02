# Cyber Security Threat Mitigation - Quick Start Guide

## 🚀 Quick Start (5 minutes)

```bash
# 1. Setup environment
cd /Users/wardento/projevt/cyber-poc
make setup

# 2. Generate small test dataset
NUM_BENIGN=10000 NUM_MALICIOUS=5000 make gen-data

# 3. Start the server (in one terminal)
make run-server

# 4. Test the API (in another terminal)
make test
```

## 📊 Generate FULL Dataset (1M records)

```bash
# This will take ~30-60 seconds
make gen-data NUM_BENIGN=700000 NUM_MALICIOUS=300000
```

## 🔍 Run Security Scans

```bash
# Scan localhost (authorized by default)
TARGET=http://localhost:8000 make recon

# The system will:
# - Check whitelist authorization
# - Run nmap, ZAP, sqlmap if available
# - Generate simulated outputs if tools missing
# - Log all results to deliverables/summary.txt
```

## 🧪 Test Detection API

```bash
# Health check
curl http://localhost:8000/health | jq

# Test benign request
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"method":"GET","url":"/search","params":{"q":"laptop"},"raw_query":"GET /search?q=laptop"}'

# Test SQL injection
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"method":"GET","url":"/product","params":{"id":"1 OR 1=1"},"raw_query":"GET /product?id=1 OR 1=1"}'

# Get statistics
curl http://localhost:8000/stats | jq

# List detection rules
curl http://localhost:8000/rules | jq
```

## 📁 Key Files & Directories

```
cyber-poc/
├── data/
│   ├── dataset.csv              # Main dataset (CSV)
│   ├── dataset.jl               # Main dataset (JSONLines)
│   ├── generation_report.json   # Generation statistics
│   └── inspection_samples.json  # Sample malicious records
├── recon/
│   ├── payloads.json            # 200+ SQL injection payloads
│   └── output/                  # Scan results
├── logs/
│   └── requests.jl              # All API requests logged here
├── deliverables/
│   └── summary.txt              # Execution log
└── backend/
    └── model_server.py          # FastAPI detection engine
```

## 🎯 Common Tasks

### View Dataset Statistics
```bash
cat data/generation_report.json | jq
```

### Inspect Malicious Samples
```bash
make inspect-samples
# Or manually:
cat data/inspection_samples.json | jq '.[0:5]'
```

### Check Server Logs
```bash
tail -f logs/requests.jl
# Or with pretty printing:
tail -f logs/requests.jl | jq
```

### View Execution Summary
```bash
make summary
# Or manually:
cat deliverables/summary.txt
```

### Clean Generated Files
```bash
# Clean logs and temporary files only
make clean

# Clean EVERYTHING including dataset
make clean-all
```

## 🔧 Configuration Variables

All targets support environment variables:

```bash
# Dataset generation
NUM_BENIGN=700000        # Number of benign records
NUM_MALICIOUS=300000     # Number of malicious records
CHUNK_SIZE=10000         # Records per chunk
AUGMENT=5                # Mutation multiplier
BOT_IP_RATIO=0.1         # Ratio of bot IPs

# Reconnaissance
TARGET=http://localhost:8000       # Target to scan
WHITE_LIST="localhost 127.0.0.1"   # Authorized targets

# Example: Custom dataset generation
make gen-data NUM_BENIGN=50000 NUM_MALICIOUS=25000 AUGMENT=3
```

## 📖 API Documentation

Once the server is running, visit:
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## ⚠️ Important Security Notes

1. **Never scan unauthorized targets**
   - Default whitelist: localhost, 127.0.0.1
   - Modify only for systems you own/have permission to test

2. **All scans are logged**
   - Unauthorized attempts logged to deliverables/recon_skipped.jl
   - Review regularly for security auditing

3. **Simulated outputs**
   - If tools (nmap/ZAP/sqlmap) unavailable, system generates realistic simulations
   - Check scan_type field: "real" vs "simulated"

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Try different port
source .venv/bin/activate
uvicorn backend.model_server:app --port 8080
```

### Dataset generation slow
```bash
# Use smaller chunk size for memory-constrained systems
make gen-data CHUNK_SIZE=5000

# Or generate smaller dataset first to test
make gen-data NUM_BENIGN=1000 NUM_MALICIOUS=500
```

### nmap permission denied
```bash
# nmap -sS requires root. Options:
# 1. Run with sudo (not recommended for this project)
# 2. Use simulated output (automatic fallback)
# 3. Use different scan type: -sT instead of -sS (modify run_nmap.py)
```

## 📊 Expected Performance

- **Dataset Generation**: ~40,000-50,000 records/second
- **Detection API**: <10ms response time per request
- **Memory Usage**: ~50-100MB base, scales with dataset in memory operations
- **Disk Space**: 
  - 1M records ≈ 500MB CSV + 1GB JSONLines
  - Payloads: ~100KB
  - Logs: Varies based on traffic

## 🚀 Next Steps (Future Enhancements)

1. **Add ML Training**
   ```bash
   # Create ml/ directory
   mkdir -p ml
   # Implement feature extraction, training, evaluation
   ```

2. **Dockerize**
   ```bash
   # Create Dockerfile and docker-compose.yml
   docker-compose up -d
   ```

3. **Add Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert notifications

4. **Scale Up**
   - Kubernetes deployment
   - Load balancing
   - Redis caching
   - PostgreSQL for logs

## 📞 Support

Check these files for more information:
- README.md - Full documentation
- PROJECT_COMPLETION_REPORT.txt - Detailed completion report
- deliverables/summary.txt - Execution history
- make help - List all available commands

---

**Remember**: This tool is for educational and authorized security testing only. 
Always obtain proper authorization before testing any system.
