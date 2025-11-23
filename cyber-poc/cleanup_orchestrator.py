#!/usr/bin/env python3
"""
Repository Cleanup Orchestrator
Performs safe, auditable cleanup with comprehensive health checks.
Default: DRY_RUN=true (simulation only)
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
import csv
import re

# ============================================================================
# CONFIGURATION
# ============================================================================
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
KEEP_MODELS = int(os.getenv("KEEP_MODELS", "3"))
MAX_AGE_DAYS_FOR_SIM = int(os.getenv("MAX_AGE_DAYS_FOR_SIM", "30"))
FORCE_DELETE_CORE = os.getenv("FORCE_DELETE_CORE", "false").lower() == "true"
CONFIRM = "--confirm" in sys.argv

REPO_ROOT = Path("/Users/wardento/projevt/cyber-poc")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = REPO_ROOT / "backups" / "cleanup" / TIMESTAMP
DELIVERABLES_DIR = REPO_ROOT / "deliverables"
LOG_FILE = DELIVERABLES_DIR / "cleanup_task_execution_log.jl"

# Protected artifacts (Safety Rule 4)
PROTECTED_PATTERNS = [
    "ml/models/best_xgboost_*.joblib",
    "ml/models/tfidf_vectorizer.joblib",
    "ml/models/model_metadata.json",
    "ml/data/merged_normalized.csv",
    "data/raw/**",
    "deliverables/**",
    "ml_final_package.zip",
]

PROTECTED_KEYWORDS = ["best", "production", "final", "deliverable", "canonical"]

# ============================================================================
# LOGGING
# ============================================================================
def log_action(task_id, action, path, result, **kwargs):
    """Log action to JSONLines file"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "action": action,
        "path": str(path),
        "dry_run": DRY_RUN,
        "result": result,
        **kwargs
    }
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return entry


# ============================================================================
# PHASE 0: ENVIRONMENT SETUP
# ============================================================================
def phase0_setup():
    """Initialize environment and validate inputs"""
    print(f"\n{'='*80}")
    print(f"PHASE 0: ENVIRONMENT SETUP")
    print(f"{'='*80}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"CONFIRM: {CONFIRM}")
    print(f"KEEP_MODELS: {KEEP_MODELS}")
    print(f"MAX_AGE_DAYS_FOR_SIM: {MAX_AGE_DAYS_FOR_SIM}")
    print(f"FORCE_DELETE_CORE: {FORCE_DELETE_CORE}")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Backup Directory: {BACKUP_DIR}")
    
    if not REPO_ROOT.exists():
        raise FileNotFoundError(f"Repository root not found: {REPO_ROOT}")
    
    # Create deliverables directory
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for project spec (may not exist, non-fatal)
    spec_path = Path("/mnt/data/project.pdf")
    spec_available = spec_path.exists()
    print(f"Project spec at /mnt/data/project.pdf: {'FOUND' if spec_available else 'NOT FOUND (optional)'}")
    
    log_action("PHASE0", "init", REPO_ROOT, "SUCCESS",
               note=f"Initialized with DRY_RUN={DRY_RUN}, KEEP_MODELS={KEEP_MODELS}")
    
    return {
        "timestamp": TIMESTAMP,
        "dry_run": DRY_RUN,
        "confirm": CONFIRM,
        "keep_models": KEEP_MODELS,
        "spec_available": spec_available
    }


# ============================================================================
# PHASE A1: FILE INVENTORY
# ============================================================================
def compute_file_hash(filepath):
    """Compute SHA256 hash of file"""
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"


def phase_a1_inventory():
    """Walk repository and create complete file inventory"""
    print(f"\n{'='*80}")
    print(f"PHASE A1: FILE INVENTORY")
    print(f"{'='*80}")
    
    inventory = []
    total_size = 0
    file_count = 0
    
    # Walk the repository
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for filename in files:
            filepath = Path(root) / filename
            try:
                stat = filepath.stat()
                rel_path = filepath.relative_to(REPO_ROOT)
                
                # Compute hash for files < 100MB
                file_hash = compute_file_hash(filepath) if stat.st_size < 100_000_000 else "SKIPPED_LARGE"
                
                file_info = {
                    "path": str(rel_path),
                    "absolute_path": str(filepath),
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                    "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "sha256": file_hash
                }
                
                inventory.append(file_info)
                total_size += stat.st_size
                file_count += 1
                
                if file_count % 100 == 0:
                    print(f"  Processed {file_count} files...", end="\r")
                    
            except Exception as e:
                print(f"  Warning: Could not process {filepath}: {e}")
                log_action("A1_INVENTORY", "scan_error", filepath, "ERROR", error=str(e))
    
    print(f"  Processed {file_count} files... DONE")
    print(f"\nInventory Summary:")
    print(f"  Total Files: {file_count:,}")
    print(f"  Total Size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
    
    # Save inventory
    inventory_file = DELIVERABLES_DIR / f"cleanup_inventory_{TIMESTAMP}.json"
    with open(inventory_file, "w") as f:
        json.dump({
            "timestamp": TIMESTAMP,
            "repo_root": str(REPO_ROOT),
            "total_files": file_count,
            "total_size_bytes": total_size,
            "files": inventory
        }, f, indent=2)
    
    print(f"\n✓ Inventory saved to: {inventory_file.relative_to(REPO_ROOT)}")
    
    log_action("A1_INVENTORY", "complete", REPO_ROOT, "SUCCESS",
               file_count=file_count, total_size_bytes=total_size)
    
    return inventory


# ============================================================================
# PHASE A2: FILE CLASSIFICATION
# ============================================================================
def is_protected(filepath):
    """Check if file is protected"""
    path_str = str(filepath).lower()
    
    # Check protected keywords
    for keyword in PROTECTED_KEYWORDS:
        if keyword in path_str:
            return True, f"Contains protected keyword: {keyword}"
    
    # Check protected patterns
    for pattern in PROTECTED_PATTERNS:
        if filepath.match(pattern.replace("**", "*")):
            return True, f"Matches protected pattern: {pattern}"
    
    # Explicit protected paths
    protected_paths = [
        "ml/models/tfidf_vectorizer.joblib",
        "ml/models/model_metadata.json",
        "ml/data/merged_normalized.csv",
    ]
    
    for ppath in protected_paths:
        if str(filepath).endswith(ppath):
            return True, f"Explicit protected path: {ppath}"
    
    return False, None


def classify_file(file_info):
    """Classify a single file"""
    filepath = Path(file_info["path"])
    size_bytes = file_info["size_bytes"]
    mtime = file_info["mtime"]
    age_days = (time.time() - mtime) / 86400
    
    # Check if protected first
    is_prot, prot_reason = is_protected(filepath)
    if is_prot:
        return "PROTECT", prot_reason
    
    # Python cache files
    if filepath.match("**/__pycache__/**") or filepath.suffix == ".pyc":
        return "SAFE_DELETE", "Python cache file"
    
    # Notebook checkpoints
    if "-checkpoint.ipynb" in str(filepath) or filepath.name.endswith(".backup.ipynb"):
        return "SAFE_DELETE", "Notebook checkpoint/backup"
    
    # macOS files
    if filepath.name == ".DS_Store":
        return "SAFE_DELETE", "macOS metadata file"
    
    # Editor temp files
    if filepath.name.endswith("~") or filepath.suffix == ".bak":
        return "SAFE_DELETE", "Editor temporary file"
    
    # Duplicate dataset copies
    if ("_copy" in filepath.name or "_backup" in filepath.name) and filepath.suffix == ".csv":
        if age_days > 30:
            return "SAFE_DELETE", f"Old duplicate dataset (age: {age_days:.0f} days)"
        else:
            return "REVIEW", f"Recent duplicate dataset (age: {age_days:.0f} days)"
    
    # Old simulation outputs
    if "recon/output" in str(filepath) and "-sim-" in filepath.name:
        if age_days > MAX_AGE_DAYS_FOR_SIM:
            return "SAFE_DELETE", f"Old simulation output (age: {age_days:.0f} days)"
    
    # Large logs
    if "logs/" in str(filepath) and size_bytes > 50_000_000:
        return "ARCHIVE", f"Large log file ({size_bytes / (1024**2):.1f} MB)"
    
    # Model artifacts (keep newest KEEP_MODELS)
    if "models/" in str(filepath) and filepath.suffix == ".joblib":
        if "best_xgboost" in filepath.name:
            return "KEEP", "Model artifact (will check if in top N)"
        else:
            return "KEEP", "Model artifact"
    
    # Large intermediate artifacts
    if size_bytes > 50_000_000 and ("shap" in str(filepath).lower() or "explainability" in str(filepath).lower()):
        return "ARCHIVE", f"Large intermediate artifact ({size_bytes / (1024**2):.1f} MB)"
    
    # Files with todo/wip
    if "todo" in filepath.name.lower() or "wip" in filepath.name.lower():
        return "REVIEW", "Work-in-progress or TODO file"
    
    # Deliverables (protected but check)
    if "deliverables/" in str(filepath):
        return "PROTECT", "Deliverable file (protected)"
    
    # Default: KEEP
    return "KEEP", "No specific rule applied"


def phase_a2_classify(inventory):
    """Classify all files in inventory"""
    print(f"\n{'='*80}")
    print(f"PHASE A2: FILE CLASSIFICATION")
    print(f"{'='*80}")
    
    classifications = {
        "PROTECT": [],
        "KEEP": [],
        "ARCHIVE": [],
        "SAFE_DELETE": [],
        "REVIEW": []
    }
    
    for file_info in inventory:
        classification, reason = classify_file(file_info)
        
        classified_entry = {
            **file_info,
            "classification": classification,
            "reason": reason
        }
        
        classifications[classification].append(classified_entry)
    
    # Special handling for model artifacts - keep only newest KEEP_MODELS
    model_files = [f for f in classifications["KEEP"] if "best_xgboost" in f["path"] and f["path"].endswith(".joblib")]
    if len(model_files) > KEEP_MODELS:
        # Sort by mtime (newest first)
        model_files.sort(key=lambda x: x["mtime"], reverse=True)
        
        # Keep newest KEEP_MODELS
        for i, model in enumerate(model_files):
            if i >= KEEP_MODELS:
                # Move to ARCHIVE
                classifications["KEEP"].remove(model)
                model["classification"] = "ARCHIVE"
                model["reason"] = f"Old model artifact (rank {i+1}, keep top {KEEP_MODELS})"
                classifications["ARCHIVE"].append(model)
    
    # Print summary
    print("\nClassification Summary:")
    for cat in ["PROTECT", "KEEP", "ARCHIVE", "SAFE_DELETE", "REVIEW"]:
        count = len(classifications[cat])
        total_size = sum(f["size_bytes"] for f in classifications[cat])
        print(f"  {cat:15s}: {count:5d} files, {total_size:15,} bytes ({total_size / (1024**2):.1f} MB)")
    
    # Save classification plan
    plan_file = DELIVERABLES_DIR / f"cleanup_plan_{TIMESTAMP}.json"
    with open(plan_file, "w") as f:
        json.dump({
            "timestamp": TIMESTAMP,
            "dry_run": DRY_RUN,
            "keep_models": KEEP_MODELS,
            "max_age_days_for_sim": MAX_AGE_DAYS_FOR_SIM,
            "classifications": classifications,
            "summary": {
                cat: {
                    "count": len(classifications[cat]),
                    "total_size_bytes": sum(f["size_bytes"] for f in classifications[cat])
                }
                for cat in classifications.keys()
            }
        }, f, indent=2)
    
    print(f"\n✓ Classification plan saved to: {plan_file.relative_to(REPO_ROOT)}")
    
    log_action("A2_CLASSIFY", "complete", REPO_ROOT, "SUCCESS",
               note=f"Classified {len(inventory)} files")
    
    return classifications


# ============================================================================
# PHASE A3: DRY-RUN REPORT
# ============================================================================
def phase_a3_report(classifications):
    """Generate human-readable dry-run report"""
    print(f"\n{'='*80}")
    print(f"PHASE A3: DRY-RUN REPORT")
    print(f"{'='*80}")
    
    report_lines = []
    report_lines.append(f"# Cleanup Plan Report - {TIMESTAMP}\n")
    report_lines.append(f"**Mode**: {'DRY-RUN (Simulation)' if DRY_RUN else 'LIVE (Destructive)'}\n")
    report_lines.append(f"**Repository**: {REPO_ROOT}\n")
    report_lines.append(f"**Parameters**:\n")
    report_lines.append(f"- KEEP_MODELS: {KEEP_MODELS}\n")
    report_lines.append(f"- MAX_AGE_DAYS_FOR_SIM: {MAX_AGE_DAYS_FOR_SIM}\n")
    report_lines.append(f"- FORCE_DELETE_CORE: {FORCE_DELETE_CORE}\n")
    report_lines.append("\n---\n\n")
    
    # Summary statistics
    report_lines.append("## Summary Statistics\n\n")
    report_lines.append("| Classification | File Count | Total Size (MB) | Total Size (GB) |\n")
    report_lines.append("|----------------|------------|-----------------|------------------|\n")
    
    for cat in ["PROTECT", "KEEP", "ARCHIVE", "SAFE_DELETE", "REVIEW"]:
        count = len(classifications[cat])
        total_size = sum(f["size_bytes"] for f in classifications[cat])
        size_mb = total_size / (1024**2)
        size_gb = total_size / (1024**3)
        report_lines.append(f"| {cat:14s} | {count:10,d} | {size_mb:15.2f} | {size_gb:16.3f} |\n")
    
    # Total recoverable space
    recoverable_size = sum(f["size_bytes"] for f in classifications["SAFE_DELETE"] + classifications["ARCHIVE"])
    report_lines.append(f"\n**Estimated Recoverable Space**: {recoverable_size:,} bytes ({recoverable_size / (1024**3):.2f} GB)\n\n")
    
    # Top 20 largest candidates for deletion/archiving
    report_lines.append("## Top 20 Largest Candidates for Deletion/Archiving\n\n")
    report_lines.append("| # | Size (MB) | Classification | Path | Reason |\n")
    report_lines.append("|---|-----------|----------------|------|--------|\n")
    
    candidates = classifications["SAFE_DELETE"] + classifications["ARCHIVE"]
    candidates.sort(key=lambda x: x["size_bytes"], reverse=True)
    
    for i, candidate in enumerate(candidates[:20], 1):
        size_mb = candidate["size_bytes"] / (1024**2)
        report_lines.append(f"| {i:2d} | {size_mb:9.2f} | {candidate['classification']:14s} | `{candidate['path']}` | {candidate['reason']} |\n")
    
    # Details by classification
    report_lines.append("\n---\n\n## Detailed File Lists\n\n")
    
    for cat in ["SAFE_DELETE", "ARCHIVE", "REVIEW"]:
        report_lines.append(f"### {cat} ({len(classifications[cat])} files)\n\n")
        
        if len(classifications[cat]) == 0:
            report_lines.append("*No files in this category*\n\n")
            continue
        
        # Group by reason
        by_reason = {}
        for f in classifications[cat]:
            reason = f["reason"]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(f)
        
        for reason, files in sorted(by_reason.items()):
            report_lines.append(f"#### {reason} ({len(files)} files)\n\n")
            total_reason_size = sum(f["size_bytes"] for f in files)
            report_lines.append(f"*Total size: {total_reason_size / (1024**2):.2f} MB*\n\n")
            
            # Show first 10 files
            for f in files[:10]:
                size_kb = f["size_bytes"] / 1024
                report_lines.append(f"- `{f['path']}` ({size_kb:.1f} KB)\n")
            
            if len(files) > 10:
                report_lines.append(f"- *... and {len(files) - 10} more files*\n")
            
            report_lines.append("\n")
    
    # Protected files summary
    report_lines.append(f"### PROTECTED ({len(classifications['PROTECT'])} files)\n\n")
    report_lines.append("*These files are protected and will never be deleted*\n\n")
    for f in sorted(classifications["PROTECT"], key=lambda x: x["path"])[:20]:
        report_lines.append(f"- `{f['path']}` - {f['reason']}\n")
    if len(classifications["PROTECT"]) > 20:
        report_lines.append(f"- *... and {len(classifications['PROTECT']) - 20} more protected files*\n")
    
    # Next steps
    report_lines.append("\n---\n\n## Next Steps\n\n")
    if DRY_RUN and not CONFIRM:
        report_lines.append("⚠️ **DRY-RUN MODE** - No files have been modified.\n\n")
        report_lines.append("To apply these changes:\n")
        report_lines.append("1. Review this report carefully\n")
        report_lines.append("2. Run with `--confirm` flag or set `DRY_RUN=false`\n")
        report_lines.append("3. Backups will be created automatically before any deletions\n")
    else:
        report_lines.append("✓ Ready to proceed with Phase B (Apply Changes) if confirmed.\n")
    
    # Save report
    report_file = DELIVERABLES_DIR / f"cleanup_plan_report_{TIMESTAMP}.md"
    with open(report_file, "w") as f:
        f.writelines(report_lines)
    
    print(f"\n✓ Dry-run report saved to: {report_file.relative_to(REPO_ROOT)}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print(f"Files to DELETE: {len(classifications['SAFE_DELETE'])}")
    print(f"Files to ARCHIVE: {len(classifications['ARCHIVE'])}")
    print(f"Files to REVIEW: {len(classifications['REVIEW'])}")
    print(f"Recoverable space: {recoverable_size / (1024**3):.2f} GB")
    print("="*80)
    
    log_action("A3_REPORT", "complete", report_file, "SUCCESS",
               recoverable_gb=recoverable_size / (1024**3))
    
    return report_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution flow"""
    try:
        # Phase 0: Setup
        config = phase0_setup()
        
        # Phase A1: Inventory
        inventory = phase_a1_inventory()
        
        # Phase A2: Classification
        classifications = phase_a2_classify(inventory)
        
        # Phase A3: Report
        report_file = phase_a3_report(classifications)
        
        print(f"\n{'='*80}")
        print("PHASE A COMPLETE")
        print(f"{'='*80}")
        print(f"\nReports generated:")
        print(f"  - Inventory: deliverables/cleanup_inventory_{TIMESTAMP}.json")
        print(f"  - Plan: deliverables/cleanup_plan_{TIMESTAMP}.json")
        print(f"  - Report: deliverables/cleanup_plan_report_{TIMESTAMP}.md")
        print(f"  - Log: deliverables/cleanup_task_execution_log.jl")
        
        if DRY_RUN and not CONFIRM:
            print(f"\n⚠️  DRY-RUN MODE - No files were modified")
            print(f"To apply changes: python cleanup_orchestrator.py --confirm")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        log_action("ERROR", "fatal", REPO_ROOT, "FAILURE", error=str(e))
        
        error_log = DELIVERABLES_DIR / "cleanup_errors.log"
        with open(error_log, "a") as f:
            f.write(f"\n[{datetime.now().isoformat()}] FATAL ERROR:\n")
            f.write(f"{str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
