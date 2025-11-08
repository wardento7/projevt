# Cleanup Plan Report

**Generated:** 2025-11-08T21:52:45.308482
**Session ID:** 20251108_215019
**Mode:** DRY RUN (no changes will be made)

---

## Executive Summary

- **Total files scanned:** 123
- **Cleanup candidates identified:** 123
- **Estimated space to reclaim:** 27.39 MB

### Breakdown by Action Type

- **safe-delete:** 11 files (3.47 MB)
- **ignore:** 112 files (23.92 MB)

---

## Detailed Analysis by Action Type

### Safe-Delete (11 files, 3.47 MB)

These files can be safely removed:

#### macOS system file (3 files)

- `.DS_Store` (10.0 KB)
- `ml/.DS_Store` (8.0 KB)
- `backend/.DS_Store` (6.0 KB)

#### Python bytecode cache (6 files)

- `ml/scripts/__pycache__/train_xgboost.cpython-312.pyc` (19.7 KB)
- `tests/__pycache__/test_api.cpython-314-pytest-8.4.2.pyc` (19.1 KB)
- `backend/__pycache__/model_server.cpython-312.pyc` (17.8 KB)
- `backend/__pycache__/model_server.cpython-314.pyc` (16.0 KB)
- `tests/__pycache__/test_detection_engine.cpython-314-pytest-8.4.2.pyc` (12.9 KB)
- `tests/__pycache__/__init__.cpython-314.pyc` (0.2 KB)

#### Backup file (2 files)

- `backups/prepare_tasks/20251108_212057/merged.csv.backup` (3428.3 KB)
- `backups/prepare_tasks/20251108_212057/model_server.py.backup` (12.3 KB)

### Ignored (112 files, 23.92 MB)

These files will NOT be touched:

- Protected core artifacts (models, datasets, configs)
- Files containing review keywords (final, deliverable, best, production)
- Files with no matching cleanup rules

---

## Top 10 Largest Candidates for Cleanup

1. `backups/prepare_tasks/20251108_212057/merged.csv.backup` - 3.35 MB (safe-delete: Backup file)
2. `ml/scripts/__pycache__/train_xgboost.cpython-312.pyc` - 0.02 MB (safe-delete: Python bytecode cache)
3. `tests/__pycache__/test_api.cpython-314-pytest-8.4.2.pyc` - 0.02 MB (safe-delete: Python bytecode cache)
4. `backend/__pycache__/model_server.cpython-312.pyc` - 0.02 MB (safe-delete: Python bytecode cache)
5. `backend/__pycache__/model_server.cpython-314.pyc` - 0.02 MB (safe-delete: Python bytecode cache)
6. `tests/__pycache__/test_detection_engine.cpython-314-pytest-8.4.2.pyc` - 0.01 MB (safe-delete: Python bytecode cache)
7. `backups/prepare_tasks/20251108_212057/model_server.py.backup` - 0.01 MB (safe-delete: Backup file)
8. `.DS_Store` - 0.01 MB (safe-delete: macOS system file)
9. `ml/.DS_Store` - 0.01 MB (safe-delete: macOS system file)
10. `backend/.DS_Store` - 0.01 MB (safe-delete: macOS system file)

---

## Safety Checks

✅ **Protected Files Verified:**

- `ml/models/best_xgboost_20251103_200539_f1_0.998.joblib` - ✓ PROTECTED
- `ml/models/tfidf_vectorizer.joblib` - ✓ PROTECTED
- `ml/models/numeric_scaler.joblib` - ✓ PROTECTED
- `ml/data/merged_normalized.csv` - ✓ PROTECTED
- `backend/models/Bestmodel.py` - ✓ PROTECTED
- `backend/model_server.py` - ✓ PROTECTED

---

## Estimated Space Reclamation

- **Safe-delete:** 3.47 MB
- **Archive (moved to backups):** 0.00 MB
- **Total space to reclaim:** 3.47 MB (0.003 GB)

*Note: Archived files are moved to `backups/cleanup/20251108_215019/` and compressed.*

---

## Next Steps

### To Proceed with Cleanup:

```bash
# Review this report carefully, then run:
# (Agent will execute with --confirm flag)
```

### What Will Happen:

1. **Backup Phase:** All files to be deleted/modified will be backed up to `backups/cleanup/20251108_215019/`
2. **Archive Phase:** Large files and old artifacts will be compressed and moved to backups
3. **Delete Phase:** Temporary files, caches, and redundant copies will be removed
4. **Verify Phase:** Integrity checks on core artifacts (model loading, syntax validation)
5. **Report Phase:** Final summary with actual results

### Safety Guarantees:

- ✅ All deletions are logged in `deliverables/cleanup_log.jl`
- ✅ Complete backups created before any changes
- ✅ Protected files cannot be deleted (safety rules enforced)
- ✅ Integrity verification after cleanup
- ✅ Rollback possible from backups if needed

---

**Report Generated:** 2025-11-08T21:52:45.308579
**Mode:** DRY RUN - No filesystem changes made
