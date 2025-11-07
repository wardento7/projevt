# Cyber-POC System Health Check Report

**Generated:** 2025-11-07T22:42:54.840360Z
**Repository:** cyber-poc/
**Total Checks:** 40

## Executive Summary

⚠️ **OVERALL STATUS: ISSUES DETECTED**

System has 2 critical failures and 8 warnings.

### Status Breakdown
- ✅ **PASS:** 29
- ⚠️ **WARNING:** 8
- ❌ **FAIL:** 2

## Component Status

### ✅ UNKNOWN
- Total checks: 1
- Passed: 0
- Warnings: 0
- Failed: 0

### ✅ REPO
- Total checks: 6
- Passed: 6
- Warnings: 0
- Failed: 0

### ✅ CYBER RECON
- Total checks: 8
- Passed: 4
- Warnings: 4
- Failed: 0

### ⚠️ ML MODEL
- Total checks: 13
- Passed: 10
- Warnings: 1
- Failed: 2

### ✅ ML REPORTS
- Total checks: 8
- Passed: 7
- Warnings: 1
- Failed: 0

### ✅ BACKEND
- Total checks: 3
- Passed: 1
- Warnings: 2
- Failed: 0

### ✅ CLEANUP
- Total checks: 1
- Passed: 1
- Warnings: 0
- Failed: 0

## Detailed Check Results

| Component | Check | Status | Path | Message |
|-----------|-------|--------|------|---------|
| repo | repo_root_exists | ✅ PASS | `/Users/wardento/projevt/cyber-poc` | Repository root is accessible |
| repo | directory_ml_exists | ✅ PASS | `ml` | Directory ml exists and accessible |
| repo | directory_recon_exists | ✅ PASS | `recon` | Directory recon exists and accessible |
| repo | directory_backend_exists | ✅ PASS | `backend` | Directory backend exists and accessible |
| repo | directory_data_exists | ✅ PASS | `data` | Directory data exists and accessible |
| repo | directory_deliverables_exists | ✅ PASS | `deliverables` | Directory deliverables exists and accessible |
| cyber_recon | payloads_json_valid | ✅ PASS | `recon/payloads.json` | Valid JSON with 200 payload entries and 1000 variants |
| cyber_recon | tool_nmap_available | ✅ PASS | `/opt/homebrew/bin/nmap` | nmap is available |
| cyber_recon | tool_sqlmap_available | ✅ PASS | `/opt/homebrew/bin/sqlmap` | sqlmap is available |
| cyber_recon | tool_zap-cli_available | ⚠️ WARNING | `NOT_FOUND` | zap-cli is not installed |
| cyber_recon | dataset_merged_valid | ⚠️ WARNING | `ml/data/merged.csv` | Dataset has 27095 rows, 19 columns, missing: ['id', 'label'] |
| cyber_recon | dataset_dataset_valid | ⚠️ WARNING | `data/dataset.csv` | Dataset has 27095 rows, 14 columns, missing: ['id', 'label'] |
| cyber_recon | dataset_FINAL_DATASET_FOR_AI_TEAM_v3 (1)_valid | ⚠️ WARNING | `FINAL_DATASET_FOR_AI_TEAM_v3 (1).csv` | Dataset has 30919 rows, 3 columns, missing: ['id', 'timestam |
| cyber_recon | recon_output_dir_exists | ✅ PASS | `recon/output` | Found 4 JSON output files |
| ml_model | artifact_model_exists | ✅ PASS | `ml/models/best_xgboost_20251103_200539_f1_0.998.jo` | model found (139.3 KB) |
| ml_model | artifact_vectorizer_exists | ✅ PASS | `ml/models/tfidf_vectorizer.joblib` | vectorizer found (169.2 KB) |
| ml_model | artifact_scaler_exists | ✅ PASS | `ml/models/numeric_scaler.joblib` | scaler found (1.3 KB) |
| ml_model | artifact_metadata_exists | ✅ PASS | `ml/models/model_metadata.json` | metadata found (0.8 KB) |
| ml_model | model_load_dependencies | ❌ FAIL | `ml/models/` | Missing required library: No module named 'joblib' |
| ml_model | model_load_success | ✅ PASS | `ml/models/` | All artifacts loaded successfully |
| ml_model | model_load_or_inference | ❌ FAIL | `ml/models/` | Error: X has 9 features, but StandardScaler is expecting 12  |
| ml_model | model_load_success | ✅ PASS | `ml/models/` | All artifacts loaded successfully |
| ml_model | inference_smoke_test | ✅ PASS | `deliverables/sample_inference.json` | Inference successful: MALICIOUS (score: 0.9965) |
| ml_model | metadata_valid | ✅ PASS | `ml/models/model_metadata.json` | Metadata valid with model: XGBoost, F1: 0.9979716024340771 |
| ml_model | thresholds_valid | ⚠️ WARNING | `ml/reports/thresholds.json` | Thresholds: block=None, challenge=None |
| ml_reports | report_model_card.md_exists | ✅ PASS | `ml/reports/model_card.md` | Found (7772 bytes) |
| ml_reports | report_dataset_report.json_exists | ✅ PASS | `ml/reports/dataset_report.json` | Found (2086 bytes) |
| ml_reports | report_model_comparison.csv_exists | ✅ PASS | `ml/reports/model_comparison.csv` | Found (224 bytes) |
| ml_reports | report_thresholds.json_exists | ✅ PASS | `ml/reports/thresholds.json` | Found (3714 bytes) |
| ml_reports | plots_directory_exists | ⚠️ WARNING | `ml/reports/plots` | Plots directory not found |
| ml_model | metadata_valid | ✅ PASS | `ml/models/model_metadata.json` | Metadata valid with model: XGBoost, F1: 0.9979716024340771 |
| ml_model | thresholds_valid | ✅ PASS | `ml/reports/thresholds.json` | Thresholds: block=None, challenge=None |
| ml_reports | report_model_card.md_exists | ✅ PASS | `ml/reports/model_card.md` | Found (7772 bytes) |
| ml_reports | report_dataset_report.json_exists | ✅ PASS | `ml/reports/dataset_report.json` | Found (2086 bytes) |
| ml_reports | report_model_comparison.csv_exists | ✅ PASS | `ml/reports/model_comparison.csv` | Found (224 bytes) |
| backend | backend_models_dir_exists | ✅ PASS | `backend/models` | Found 4 artifacts |
| backend | model_server_endpoints | ⚠️ WARNING | `backend/model_server.py` | Endpoints: /infer=True, /infer-ml=False |
| backend | bestmodel_module_exists | ⚠️ WARNING | `backend/models/Bestmodel.py` | Module not found |
| cleanup | cleanup_candidates_identified | ✅ PASS | `deliverables/cleanup_candidates.csv` | Found 0 cleanup candidates |

## Sample Outputs

### ML Model Inference Test
```json
{
  "sample_query": "SELECT * FROM users WHERE id=1 OR 1=1 --",
  "prediction": 1,
  "prediction_label": "MALICIOUS",
  "malicious_probability": 0.9964851140975952,
  "benign_probability": 0.003514885902404785,
  "recommended_action": "BLOCK",
  "features": {
    "length": 40,
    "single_quotes": 0,
    "sql_keywords": 1,
    "has_or_equals": true
  },
  "timestamp": "2025-11-07T22:39:46.795462Z"
}
```

## Key File Locations

### ML Models
- Model: `ml/models/best_xgboost_20251103_200539_f1_0.998.joblib`
- Vectorizer: `ml/models/tfidf_vectorizer.joblib`
- Scaler: `ml/models/numeric_scaler.joblib`
- Metadata: `ml/models/model_metadata.json`

### Backend
- Models: `backend/models/`
- Server: `backend/model_server.py`

### Recon/Cyber
- Payloads: `recon/payloads.json` (200 entries, 1200 total payloads)
- Output: `recon/output/`

### Datasets
- ML Data: `ml/data/merged.csv` (27,095 rows)
- Raw Data: `data/dataset.csv` (27,095 rows)

## Issues and Remediation

### ❌ Critical Issues (FAIL)

**model_load_dependencies**
- Path: `ml/models/`
- Issue: Missing required library: No module named 'joblib'
- Fix: Install required packages: pip install joblib numpy scipy scikit-learn xgboost
- Effort: Quick (< 30 min)

**model_load_or_inference**
- Path: `ml/models/`
- Issue: Error: X has 9 features, but StandardScaler is expecting 12 features as input.
- Fix: Check model compatibility and feature pipeline
- Effort: Quick (< 30 min)

### ⚠️ Warnings

**tool_zap-cli_available**
- Path: `NOT_FOUND`
- Issue: zap-cli is not installed
- Fix: Install zap-cli for full recon capabilities

**dataset_merged_valid**
- Path: `ml/data/merged.csv`
- Issue: Dataset has 27095 rows, 19 columns, missing: ['id', 'label']
- Fix: Add missing fields: ['id', 'label']

**dataset_dataset_valid**
- Path: `data/dataset.csv`
- Issue: Dataset has 27095 rows, 14 columns, missing: ['id', 'label']
- Fix: Add missing fields: ['id', 'label']

**dataset_FINAL_DATASET_FOR_AI_TEAM_v3 (1)_valid**
- Path: `FINAL_DATASET_FOR_AI_TEAM_v3 (1).csv`
- Issue: Dataset has 30919 rows, 3 columns, missing: ['id', 'timestamp', 'source_ip', 'method', 'url', 'params', 'body', 'raw_query', 'label', 'attack_type']
- Fix: Add missing fields: ['id', 'timestamp', 'source_ip', 'method', 'url', 'params', 'body', 'raw_query', 'label', 'attack_type']

**thresholds_valid**
- Path: `ml/reports/thresholds.json`
- Issue: Thresholds: block=None, challenge=None

## Commands Executed

All checks were non-destructive:
```bash
# Repository structure check
du -sh ml/ recon/ backend/ data/ deliverables/

# Model loading and inference test
python -c "import joblib; model = joblib.load('ml/models/best_xgboost...'); ..."

# Tool availability checks
which nmap sqlmap zap-cli

# Payload validation
python -c "import json; payloads = json.load(open('recon/payloads.json'))"
```

## Artifacts Generated

- `deliverables/system_check_report.md` - This report
- `deliverables/system_check_summary.json` - Machine-readable summary
- `deliverables/system_check_summary.csv` - CSV format summary
- `deliverables/system_check_runlog.jl` - JSONLines execution log
- `deliverables/sample_inference.json` - ML inference test result
- `deliverables/cleanup_candidates.csv` - Cleanup recommendations
- `ml_final_package/validation/sample_prediction.txt` - Inference output

---
*Report generated on 2025-11-07T22:42:54.840360Z by non-destructive health check system*