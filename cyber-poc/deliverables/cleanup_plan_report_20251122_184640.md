# Cleanup Plan Report - 20251122_184640
**Mode**: DRY-RUN (Simulation)
**Repository**: /Users/wardento/projevt/cyber-poc
**Parameters**:
- KEEP_MODELS: 3
- MAX_AGE_DAYS_FOR_SIM: 30
- FORCE_DELETE_CORE: False

---

## Summary Statistics

| Classification | File Count | Total Size (MB) | Total Size (GB) |
|----------------|------------|-----------------|------------------|
| PROTECT        |         62 |           11.94 |            0.012 |
| KEEP           |     11,073 |          401.43 |            0.392 |
| ARCHIVE        |          0 |            0.00 |            0.000 |
| SAFE_DELETE    |          1 |            0.01 |            0.000 |
| REVIEW         |          0 |            0.00 |            0.000 |

**Estimated Recoverable Space**: 11,699 bytes (0.00 GB)

## Top 20 Largest Candidates for Deletion/Archiving

| # | Size (MB) | Classification | Path | Reason |
|---|-----------|----------------|------|--------|
|  1 |      0.01 | SAFE_DELETE    | `.venv/lib/python3.14/site-packages/_distutils_hack/__pycache__/__init__.cpython-314.pyc` | Python cache file |

---

## Detailed File Lists

### SAFE_DELETE (1 files)

#### Python cache file (1 files)

*Total size: 0.01 MB*

- `.venv/lib/python3.14/site-packages/_distutils_hack/__pycache__/__init__.cpython-314.pyc` (11.4 KB)

### ARCHIVE (0 files)

*No files in this category*

### REVIEW (0 files)

*No files in this category*

### PROTECTED (62 files)

*These files are protected and will never be deleted*

- `.venv/lib/python3.14/site-packages/pandas/tests/generic/test_finalize.py` - Contains protected keyword: final
- `.venv/lib/python3.14/site-packages/pip/_vendor/urllib3/packages/backports/weakref_finalize.py` - Contains protected keyword: final
- `.venv/lib/python3.14/site-packages/scipy/optimize/_trustregion_constr/canonical_constraint.py` - Contains protected keyword: canonical
- `.venv/lib/python3.14/site-packages/scipy/optimize/_trustregion_constr/tests/test_canonical_constraint.py` - Contains protected keyword: canonical
- `FINAL_DATASET_FOR_AI_TEAM_v3 (1).csv` - Contains protected keyword: final
- `backend/models/Bestmodel.py` - Contains protected keyword: best
- `backend/models/__pycache__/Bestmodel.cpython-314.pyc` - Contains protected keyword: best
- `backend/models/best_xgboost_20251103_200539_f1_0.998.joblib` - Contains protected keyword: best
- `backups/cleanup/20251108_220000/ml_final_package/COMPLETION_CHECKLIST.md` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/README_DELIVERY.md` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/data/merged.csv` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/models/best_xgboost_20251103_200539_f1_0.998.joblib` - Contains protected keyword: best
- `backups/cleanup/20251108_220000/ml_final_package/models/model_metadata.json` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/models/numeric_scaler.joblib` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/models/tfidf_vectorizer.joblib` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/reports/dataset_report.json` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/reports/model_card.md` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/reports/model_comparison.csv` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/reports/plots/accuracy.png` - Contains protected keyword: final
- `backups/cleanup/20251108_220000/ml_final_package/reports/plots/f1.png` - Contains protected keyword: final
- *... and 42 more protected files*

---

## Next Steps

⚠️ **DRY-RUN MODE** - No files have been modified.

To apply these changes:
1. Review this report carefully
2. Run with `--confirm` flag or set `DRY_RUN=false`
3. Backups will be created automatically before any deletions
