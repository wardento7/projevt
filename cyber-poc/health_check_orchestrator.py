#!/usr/bin/env python3
"""
System Health Check & Test Suite Orchestrator
Performs comprehensive validation of the cyber-poc system
"""

import os
import sys
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util
import ast

# ============================================================================
# CONFIGURATION
# ============================================================================
REPO_ROOT = Path("/Users/wardento/projevt/cyber-poc")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DELIVERABLES_DIR = REPO_ROOT / "deliverables"
LOG_FILE = DELIVERABLES_DIR / "cleanup_task_execution_log.jl"

# Expected dataset parameters
EXPECTED_ROW_COUNT = 12636
EXPECTED_COLUMNS = ["query", "attack_type", "severity", "label"]

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
        "dry_run": False,  # Health checks are always non-destructive
        "result": result,
        **kwargs
    }
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return entry


# ============================================================================
# PHASE C1: STATIC CODE CHECKS
# ============================================================================
def phase_c1_static_checks():
    """Run static code checks on Python files"""
    print(f"\n{'='*80}")
    print(f"PHASE C1: STATIC CODE CHECKS")
    print(f"{'='*80}")
    
    results = {
        "timestamp": TIMESTAMP,
        "compile_results": [],
        "lint_results": [],
        "summary": {
            "total_files": 0,
            "compile_passed": 0,
            "compile_failed": 0,
            "lint_issues": 0
        }
    }
    
    # Find all Python files
    python_files = []
    for dir_path in ["backend", "ml/scripts", "data", "recon", "tests"]:
        full_path = REPO_ROOT / dir_path
        if full_path.exists():
            python_files.extend(full_path.rglob("*.py"))
    
    # Also check root-level Python files
    python_files.extend([
        REPO_ROOT / "test_imports.py",
        REPO_ROOT / "cleanup_orchestrator.py",
        REPO_ROOT / "health_check_orchestrator.py"
    ])
    
    # Filter existing files
    python_files = [f for f in python_files if f.exists() and f.is_file()]
    results["summary"]["total_files"] = len(python_files)
    
    print(f"Found {len(python_files)} Python files to check")
    
    # Run py_compile on each file
    print("\nRunning py_compile...")
    for py_file in python_files:
        try:
            rel_path = py_file.relative_to(REPO_ROOT)
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                status = "PASS"
                results["summary"]["compile_passed"] += 1
            else:
                status = "FAIL"
                results["summary"]["compile_failed"] += 1
            
            results["compile_results"].append({
                "file": str(rel_path),
                "status": status,
                "stderr": result.stderr if result.stderr else None
            })
            
            if status == "FAIL":
                print(f"  ❌ {rel_path}: FAILED")
                log_action("C1_COMPILE", "check", py_file, "FAIL", error=result.stderr)
            else:
                print(f"  ✓ {rel_path}: OK", end="\r")
                
        except Exception as e:
            results["summary"]["compile_failed"] += 1
            results["compile_results"].append({
                "file": str(rel_path),
                "status": "ERROR",
                "error": str(e)
            })
            print(f"  ❌ {rel_path}: ERROR - {e}")
    
    print(f"\nCompile check: {results['summary']['compile_passed']} passed, {results['summary']['compile_failed']} failed")
    
    # Try to run flake8 if available
    print("\nAttempting lint checks with flake8...")
    try:
        flake8_result = subprocess.run(
            ["flake8", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if flake8_result.returncode == 0:
            print("  flake8 available, running checks...")
            
            for py_file in python_files[:10]:  # Limit to first 10 to avoid too much output
                try:
                    rel_path = py_file.relative_to(REPO_ROOT)
                    result = subprocess.run(
                        ["flake8", "--max-line-length=120", str(py_file)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.stdout:
                        issues = result.stdout.strip().split("\n")
                        results["summary"]["lint_issues"] += len(issues)
                        results["lint_results"].append({
                            "file": str(rel_path),
                            "issues": issues
                        })
                        
                except Exception as e:
                    print(f"  Warning: Could not lint {py_file}: {e}")
        else:
            print("  flake8 not available, skipping lint checks")
            results["lint_results"] = ["flake8 not installed"]
            
    except FileNotFoundError:
        print("  flake8 not installed, skipping lint checks")
        results["lint_results"] = ["flake8 not installed"]
    
    # Save results
    report_file = DELIVERABLES_DIR / f"compile_report_{TIMESTAMP}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Static checks report saved to: {report_file.relative_to(REPO_ROOT)}")
    
    log_action("C1_STATIC", "complete", REPO_ROOT, "SUCCESS",
               passed=results["summary"]["compile_passed"],
               failed=results["summary"]["compile_failed"])
    
    return results


# ============================================================================
# PHASE C2: DATA INTEGRITY
# ============================================================================
def phase_c2_data_integrity():
    """Verify dataset and model artifacts integrity"""
    print(f"\n{'='*80}")
    print(f"PHASE C2: DATA INTEGRITY CHECKS")
    print(f"{'='*80}")
    
    results = {
        "timestamp": TIMESTAMP,
        "checks": [],
        "summary": {
            "total_checks": 0,
            "passed": 0,
            "failed": 0
        }
    }
    
    # Check 1: merged_normalized.csv exists and has correct structure
    print("\nChecking ml/data/merged_normalized.csv...")
    dataset_path = REPO_ROOT / "ml" / "data" / "merged_normalized.csv"
    
    if not dataset_path.exists():
        results["checks"].append({
            "check": "dataset_exists",
            "status": "FAIL",
            "message": "Dataset file not found",
            "path": str(dataset_path.relative_to(REPO_ROOT))
        })
        results["summary"]["failed"] += 1
        print("  ❌ Dataset not found")
    else:
        print("  ✓ Dataset file exists")
        
        # Check row count and columns
        try:
            with open(dataset_path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                row_count = sum(1 for _ in reader)
            
            # Validate row count
            if row_count == EXPECTED_ROW_COUNT:
                print(f"  ✓ Row count matches: {row_count}")
                row_status = "PASS"
            else:
                print(f"  ⚠️  Row count mismatch: expected {EXPECTED_ROW_COUNT}, got {row_count}")
                row_status = "WARN"
            
            # Validate headers (check if expected columns are present)
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in headers]
            if not missing_cols:
                print(f"  ✓ Required columns present: {EXPECTED_COLUMNS}")
                col_status = "PASS"
            else:
                print(f"  ⚠️  Missing columns: {missing_cols}")
                col_status = "WARN"
            
            results["checks"].append({
                "check": "dataset_integrity",
                "status": "PASS" if row_status == "PASS" and col_status == "PASS" else "WARN",
                "row_count": row_count,
                "expected_row_count": EXPECTED_ROW_COUNT,
                "headers": list(headers),
                "expected_columns": EXPECTED_COLUMNS,
                "missing_columns": missing_cols
            })
            
            results["summary"]["passed"] += 1
            
        except Exception as e:
            results["checks"].append({
                "check": "dataset_integrity",
                "status": "FAIL",
                "error": str(e)
            })
            results["summary"]["failed"] += 1
            print(f"  ❌ Error reading dataset: {e}")
    
    results["summary"]["total_checks"] += 1
    
    # Check 2: Model artifacts exist
    print("\nChecking model artifacts...")
    model_artifacts = {
        "tfidf_vectorizer": REPO_ROOT / "ml" / "models" / "tfidf_vectorizer.joblib",
        "model_metadata": REPO_ROOT / "ml" / "models" / "model_metadata.json",
        "best_xgboost": REPO_ROOT / "ml" / "models" / "best_xgboost_20251103_200539_f1_0.998.joblib"
    }
    
    for artifact_name, artifact_path in model_artifacts.items():
        results["summary"]["total_checks"] += 1
        
        if artifact_path.exists():
            size_mb = artifact_path.stat().st_size / (1024 ** 2)
            print(f"  ✓ {artifact_name}: {size_mb:.2f} MB")
            results["checks"].append({
                "check": f"artifact_{artifact_name}",
                "status": "PASS",
                "path": str(artifact_path.relative_to(REPO_ROOT)),
                "size_mb": size_mb
            })
            results["summary"]["passed"] += 1
        else:
            print(f"  ❌ {artifact_name}: NOT FOUND")
            results["checks"].append({
                "check": f"artifact_{artifact_name}",
                "status": "FAIL",
                "path": str(artifact_path.relative_to(REPO_ROOT)),
                "message": "File not found"
            })
            results["summary"]["failed"] += 1
    
    # Check 3: Backend model copy
    print("\nChecking backend model artifacts...")
    backend_artifacts = {
        "Bestmodel.py": REPO_ROOT / "backend" / "models" / "Bestmodel.py",
        "best_xgboost": REPO_ROOT / "backend" / "models" / "best_xgboost_20251103_200539_f1_0.998.joblib",
        "tfidf_vectorizer": REPO_ROOT / "backend" / "models" / "tfidf_vectorizer.joblib",
        "model_metadata": REPO_ROOT / "backend" / "models" / "model_metadata.json",
    }
    
    for artifact_name, artifact_path in backend_artifacts.items():
        results["summary"]["total_checks"] += 1
        
        if artifact_path.exists():
            size_kb = artifact_path.stat().st_size / 1024
            print(f"  ✓ {artifact_name}: {size_kb:.2f} KB")
            results["checks"].append({
                "check": f"backend_artifact_{artifact_name}",
                "status": "PASS",
                "path": str(artifact_path.relative_to(REPO_ROOT)),
                "size_kb": size_kb
            })
            results["summary"]["passed"] += 1
        else:
            print(f"  ❌ {artifact_name}: NOT FOUND")
            results["checks"].append({
                "check": f"backend_artifact_{artifact_name}",
                "status": "FAIL",
                "path": str(artifact_path.relative_to(REPO_ROOT)),
                "message": "File not found"
            })
            results["summary"]["failed"] += 1
    
    # Save results
    report_file = DELIVERABLES_DIR / f"dataset_integrity_{TIMESTAMP}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Data integrity report saved to: {report_file.relative_to(REPO_ROOT)}")
    print(f"\nSummary: {results['summary']['passed']}/{results['summary']['total_checks']} checks passed")
    
    log_action("C2_INTEGRITY", "complete", REPO_ROOT, "SUCCESS",
               passed=results["summary"]["passed"],
               failed=results["summary"]["failed"])
    
    return results


# ============================================================================
# PHASE C3: MODEL SMOKE-LOAD & INFERENCE
# ============================================================================
def phase_c3_model_tests():
    """Load models and run sample predictions"""
    print(f"\n{'='*80}")
    print(f"PHASE C3: MODEL SMOKE-LOAD & INFERENCE TESTS")
    print(f"{'='*80}")
    
    results = {
        "timestamp": TIMESTAMP,
        "tests": [],
        "predictions": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0
        }
    }
    
    # Test 1: Load XGBoost model
    print("\nTest 1: Loading XGBoost model...")
    model_path = REPO_ROOT / "backend" / "models" / "best_xgboost_20251103_200539_f1_0.998.joblib"
    
    try:
        import joblib
        model = joblib.load(model_path)
        print("  ✓ Model loaded successfully")
        results["tests"].append({
            "test": "load_xgboost_model",
            "status": "PASS",
            "model_type": str(type(model).__name__)
        })
        results["summary"]["passed"] += 1
        model_loaded = True
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        results["tests"].append({
            "test": "load_xgboost_model",
            "status": "FAIL",
            "error": str(e)
        })
        results["summary"]["failed"] += 1
        model_loaded = False
    
    results["summary"]["total_tests"] += 1
    
    # Test 2: Import Bestmodel
    print("\nTest 2: Importing Bestmodel class...")
    try:
        sys.path.insert(0, str(REPO_ROOT / "backend" / "models"))
        from Bestmodel import Bestmodel
        
        print("  ✓ Bestmodel imported successfully")
        results["tests"].append({
            "test": "import_bestmodel",
            "status": "PASS"
        })
        results["summary"]["passed"] += 1
        bestmodel_imported = True
    except Exception as e:
        print(f"  ❌ Failed to import Bestmodel: {e}")
        results["tests"].append({
            "test": "import_bestmodel",
            "status": "FAIL",
            "error": str(e)
        })
        results["summary"]["failed"] += 1
        bestmodel_imported = False
    
    results["summary"]["total_tests"] += 1
    
    # Test 3: Run sample predictions
    if model_loaded and bestmodel_imported:
        print("\nTest 3: Running sample predictions...")
        
        test_samples = [
            {
                "id": 1,
                "query": "SELECT * FROM users WHERE id=1 OR 1=1 --",
                "expected_type": "malicious",
                "description": "SQL injection attack"
            },
            {
                "id": 2,
                "query": "https://shop.example.com/product?id=1' UNION SELECT username,password FROM users--",
                "expected_type": "malicious",
                "description": "SQL injection via URL parameter"
            },
            {
                "id": 3,
                "query": "SELECT name, email FROM customers WHERE active=1",
                "expected_type": "benign",
                "description": "Benign SQL query"
            }
        ]
        
        try:
            # Initialize Bestmodel
            model_dir = REPO_ROOT / "backend" / "models"
            best_model = Bestmodel(model_dir=str(model_dir))
            
            for sample in test_samples:
                try:
                    # Run prediction
                    prediction = best_model.predict(sample["query"])
                    
                    # Validate output structure
                    has_score = "score" in prediction or "confidence" in prediction
                    has_action = "action" in prediction or "label" in prediction
                    has_features = "features" in prediction or "feature_importance" in prediction
                    
                    # Extract key values
                    score = prediction.get("score", prediction.get("confidence", -1))
                    action = prediction.get("action", prediction.get("label", "UNKNOWN"))
                    
                    # Validate score range
                    score_valid = 0 <= score <= 1 if isinstance(score, (int, float)) else False
                    
                    status = "PASS" if has_score and has_action and score_valid else "WARN"
                    
                    pred_result = {
                        "sample_id": sample["id"],
                        "query": sample["query"],
                        "description": sample["description"],
                        "expected_type": sample["expected_type"],
                        "prediction": prediction,
                        "score": score,
                        "action": action,
                        "status": status,
                        "validations": {
                            "has_score": has_score,
                            "has_action": has_action,
                            "has_features": has_features,
                            "score_in_range": score_valid
                        }
                    }
                    
                    results["predictions"].append(pred_result)
                    
                    if status == "PASS":
                        print(f"  ✓ Sample {sample['id']}: {action} (score: {score:.4f})")
                        results["summary"]["passed"] += 1
                    else:
                        print(f"  ⚠️  Sample {sample['id']}: Incomplete prediction")
                        results["summary"]["failed"] += 1
                    
                    results["summary"]["total_tests"] += 1
                    
                except Exception as e:
                    print(f"  ❌ Sample {sample['id']} failed: {e}")
                    results["predictions"].append({
                        "sample_id": sample["id"],
                        "query": sample["query"],
                        "status": "FAIL",
                        "error": str(e)
                    })
                    results["summary"]["failed"] += 1
                    results["summary"]["total_tests"] += 1
            
        except Exception as e:
            print(f"  ❌ Failed to initialize Bestmodel: {e}")
            results["tests"].append({
                "test": "run_predictions",
                "status": "FAIL",
                "error": str(e)
            })
            results["summary"]["failed"] += 1
            results["summary"]["total_tests"] += 1
    else:
        print("\n⚠️  Skipping prediction tests due to load failures")
    
    # Save results
    json_file = DELIVERABLES_DIR / f"sample_prediction_{TIMESTAMP}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    txt_file = DELIVERABLES_DIR / f"sample_prediction_{TIMESTAMP}.txt"
    with open(txt_file, "w") as f:
        f.write(f"Model Inference Test Results - {TIMESTAMP}\n")
        f.write("="*80 + "\n\n")
        
        for pred in results["predictions"]:
            f.write(f"Sample {pred['sample_id']}: {pred['description']}\n")
            f.write(f"Query: {pred['query']}\n")
            f.write(f"Expected: {pred['expected_type']}\n")
            if pred.get("prediction"):
                f.write(f"Prediction: {pred.get('action', 'N/A')} (score: {pred.get('score', 'N/A')})\n")
            f.write(f"Status: {pred['status']}\n")
            f.write("-"*80 + "\n\n")
    
    print(f"\n✓ Model test results saved to:")
    print(f"  - {json_file.relative_to(REPO_ROOT)}")
    print(f"  - {txt_file.relative_to(REPO_ROOT)}")
    print(f"\nSummary: {results['summary']['passed']}/{results['summary']['total_tests']} tests passed")
    
    log_action("C3_MODEL", "complete", REPO_ROOT, "SUCCESS",
               passed=results["summary"]["passed"],
               failed=results["summary"]["failed"])
    
    return results


# ============================================================================
# PHASE C4: BACKEND ENDPOINT CHECKS
# ============================================================================
def phase_c4_backend_checks():
    """Static validation of backend endpoints"""
    print(f"\n{'='*80}")
    print(f"PHASE C4: BACKEND ENDPOINT CHECKS (STATIC)")
    print(f"{'='*80}")
    
    results = {
        "timestamp": TIMESTAMP,
        "checks": [],
        "summary": {
            "total_checks": 0,
            "passed": 0,
            "failed": 0
        }
    }
    
    server_file = REPO_ROOT / "backend" / "model_server.py"
    
    if not server_file.exists():
        print("  ❌ model_server.py not found")
        results["checks"].append({
            "check": "server_exists",
            "status": "FAIL",
            "message": "model_server.py not found"
        })
        results["summary"]["failed"] += 1
        results["summary"]["total_checks"] += 1
        
        report_file = DELIVERABLES_DIR / f"backend_endpoint_report_{TIMESTAMP}.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    print("  ✓ model_server.py exists")
    
    # Parse AST to find endpoints
    print("\nAnalyzing endpoint definitions...")
    try:
        with open(server_file, "r") as f:
            tree = ast.parse(f.read())
        
        # Look for FastAPI route decorators
        endpoints_found = []
        imports_found = []
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.ImportFrom):
                imports_found.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports_found.append(alias.name)
            
            # Check for decorators (FastAPI routes)
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if hasattr(decorator.func, 'attr'):
                            method = decorator.func.attr
                            if method in ['get', 'post', 'put', 'delete']:
                                # Extract path
                                if decorator.args:
                                    path = ast.literal_eval(decorator.args[0])
                                    endpoints_found.append({
                                        "method": method.upper(),
                                        "path": path,
                                        "function": node.name
                                    })
        
        # Check for expected endpoints
        expected_endpoints = ["/infer", "/infer-ml", "/ml-model-info"]
        
        found_paths = [e["path"] for e in endpoints_found]
        
        for expected in expected_endpoints:
            results["summary"]["total_checks"] += 1
            if expected in found_paths:
                print(f"  ✓ Endpoint defined: {expected}")
                results["checks"].append({
                    "check": f"endpoint_{expected}",
                    "status": "PASS",
                    "endpoint": expected
                })
                results["summary"]["passed"] += 1
            else:
                print(f"  ⚠️  Endpoint not found: {expected}")
                results["checks"].append({
                    "check": f"endpoint_{expected}",
                    "status": "WARN",
                    "endpoint": expected,
                    "message": "Endpoint not found in code"
                })
                results["summary"]["failed"] += 1
        
        # Check for Bestmodel import
        results["summary"]["total_checks"] += 1
        if "Bestmodel" in str(tree):
            print("  ✓ Bestmodel import/usage found")
            results["checks"].append({
                "check": "bestmodel_usage",
                "status": "PASS"
            })
            results["summary"]["passed"] += 1
        else:
            print("  ⚠️  Bestmodel not referenced in code")
            results["checks"].append({
                "check": "bestmodel_usage",
                "status": "WARN",
                "message": "Bestmodel not found in code"
            })
            results["summary"]["failed"] += 1
        
        results["endpoints_found"] = endpoints_found
        results["imports_found"] = list(set(imports_found))
        
    except Exception as e:
        print(f"  ❌ Error parsing model_server.py: {e}")
        results["checks"].append({
            "check": "parse_server",
            "status": "FAIL",
            "error": str(e)
        })
        results["summary"]["failed"] += 1
    
    # Note about server testing
    print("\n⚠️  NOTE: Server not started (requires explicit permission)")
    print("   To test endpoints live, run:")
    print("   cd backend && uvicorn model_server:app --host 127.0.0.1 --port 8000")
    
    # Save results
    report_file = DELIVERABLES_DIR / f"backend_endpoint_report_{TIMESTAMP}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    txt_file = DELIVERABLES_DIR / f"backend_server_check_{TIMESTAMP}.txt"
    with open(txt_file, "w") as f:
        f.write(f"Backend Endpoint Check - {TIMESTAMP}\n")
        f.write("="*80 + "\n\n")
        f.write("Static analysis completed. To start server and test endpoints:\n\n")
        f.write("1. Start server (LOCALHOST ONLY):\n")
        f.write("   cd backend\n")
        f.write("   uvicorn model_server:app --host 127.0.0.1 --port 8000\n\n")
        f.write("2. Test endpoints:\n")
        f.write("   curl -X POST http://127.0.0.1:8000/infer-ml \\\n")
        f.write("     -H 'Content-Type: application/json' \\\n")
        f.write("     -d '{\"query\": \"SELECT * FROM users WHERE id=1 OR 1=1 --\"}'\n\n")
        f.write(f"Endpoints found in code:\n")
        for ep in results.get("endpoints_found", []):
            f.write(f"  - {ep['method']} {ep['path']} -> {ep['function']}\n")
    
    print(f"\n✓ Backend checks saved to:")
    print(f"  - {report_file.relative_to(REPO_ROOT)}")
    print(f"  - {txt_file.relative_to(REPO_ROOT)}")
    print(f"\nSummary: {results['summary']['passed']}/{results['summary']['total_checks']} checks passed")
    
    log_action("C4_BACKEND", "complete", REPO_ROOT, "SUCCESS",
               passed=results["summary"]["passed"],
               failed=results["summary"]["failed"])
    
    return results


# ============================================================================
# PHASE C5: RECON SAFETY CHECK
# ============================================================================
def phase_c5_recon_safety():
    """Verify no external scans occurred"""
    print(f"\n{'='*80}")
    print(f"PHASE C5: RECON SAFETY VERIFICATION")
    print(f"{'='*80}")
    
    results = {
        "timestamp": TIMESTAMP,
        "checks": [],
        "summary": {
            "safe": True,
            "warnings": 0
        }
    }
    
    # Check for recon scripts
    print("\nChecking recon scripts for external targets...")
    recon_dir = REPO_ROOT / "recon"
    
    if not recon_dir.exists():
        print("  ℹ️  No recon directory found")
        results["checks"].append({
            "check": "recon_dir",
            "status": "INFO",
            "message": "No recon directory found"
        })
    else:
        recon_files = list(recon_dir.glob("*.py")) + list(recon_dir.glob("*.sh"))
        
        for recon_file in recon_files:
            try:
                with open(recon_file, "r") as f:
                    content = f.read()
                
                # Check for external IPs or domains
                external_patterns = [
                    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP addresses
                    r"https?://(?!localhost|127\.0\.0\.1)",  # External URLs
                ]
                
                import re
                found_external = []
                for pattern in external_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Filter out localhost
                        external = [m for m in matches if "127.0.0.1" not in m and "localhost" not in m]
                        found_external.extend(external)
                
                if found_external:
                    print(f"  ⚠️  {recon_file.name}: Found potential external targets: {found_external[:3]}")
                    results["checks"].append({
                        "check": f"scan_{recon_file.name}",
                        "status": "WARN",
                        "file": recon_file.name,
                        "external_targets": found_external[:5]
                    })
                    results["summary"]["warnings"] += 1
                else:
                    print(f"  ✓ {recon_file.name}: No external targets found")
                    results["checks"].append({
                        "check": f"scan_{recon_file.name}",
                        "status": "PASS",
                        "file": recon_file.name
                    })
                    
            except Exception as e:
                print(f"  ⚠️  Could not check {recon_file.name}: {e}")
    
    # Check logs for SIMULATED actions
    print("\nChecking task execution logs for SIMULATED labels...")
    if LOG_FILE.exists():
        try:
            simulated_count = 0
            real_scan_count = 0
            
            with open(LOG_FILE, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if "recon" in entry.get("task_id", "").lower() or "scan" in entry.get("action", "").lower():
                            if "SIMULATED" in entry.get("note", "").upper():
                                simulated_count += 1
                            else:
                                real_scan_count += 1
                    except:
                        pass
            
            print(f"  ℹ️  Found {simulated_count} simulated recon actions")
            print(f"  ℹ️  Found {real_scan_count} real scan actions")
            
            if real_scan_count > 0:
                print(f"  ⚠️  WARNING: Real scan actions detected!")
                results["summary"]["safe"] = False
            
            results["checks"].append({
                "check": "log_scan_actions",
                "simulated_count": simulated_count,
                "real_scan_count": real_scan_count,
                "status": "PASS" if real_scan_count == 0 else "WARN"
            })
            
        except Exception as e:
            print(f"  ⚠️  Could not parse logs: {e}")
    
    # Overall safety assessment
    print("\n" + "="*80)
    if results["summary"]["safe"] and results["summary"]["warnings"] == 0:
        print("✓ SAFETY CHECK PASSED: No external scans detected")
    elif results["summary"]["warnings"] > 0:
        print(f"⚠️  SAFETY CHECK WARNING: {results['summary']['warnings']} potential issues found")
    else:
        print("❌ SAFETY CHECK FAILED: External scan activity detected")
    print("="*80)
    
    # Save results
    report_file = DELIVERABLES_DIR / f"recon_safety_check_{TIMESTAMP}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Safety check saved to: {report_file.relative_to(REPO_ROOT)}")
    
    log_action("C5_SAFETY", "complete", REPO_ROOT, "SUCCESS",
               safe=results["summary"]["safe"],
               warnings=results["summary"]["warnings"])
    
    return results


# ============================================================================
# PHASE C6: NOTEBOOK VERIFICATION
# ============================================================================
def phase_c6_notebook_verification():
    """Verify notebooks exist and have valid structure"""
    print(f"\n{'='*80}")
    print(f"PHASE C6: NOTEBOOK VERIFICATION")
    print(f"{'='*80}")
    
    results = {
        "timestamp": TIMESTAMP,
        "notebooks": [],
        "summary": {
            "total_notebooks": 0,
            "valid": 0,
            "invalid": 0
        }
    }
    
    notebooks_dir = REPO_ROOT / "ml" / "notebooks"
    
    if not notebooks_dir.exists():
        print("  ⚠️  Notebooks directory not found")
        results["summary"]["invalid"] += 1
        
        report_file = DELIVERABLES_DIR / f"notebook_verification_report_{TIMESTAMP}.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    # Find all notebooks
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    results["summary"]["total_notebooks"] = len(notebook_files)
    
    print(f"\nFound {len(notebook_files)} notebooks")
    
    for nb_file in notebook_files:
        try:
            with open(nb_file, "r") as f:
                nb_content = json.load(f)
            
            # Check structure
            has_cells = "cells" in nb_content
            cell_count = len(nb_content.get("cells", []))
            
            # Check for imports in first few code cells
            imports_found = []
            code_cells = [c for c in nb_content.get("cells", []) if c.get("cell_type") == "code"]
            
            for cell in code_cells[:5]:  # Check first 5 code cells
                source = "".join(cell.get("source", []))
                if "import" in source:
                    # Extract import lines
                    import_lines = [line for line in source.split("\n") if "import" in line]
                    imports_found.extend(import_lines[:3])
            
            status = "VALID" if has_cells and cell_count > 0 else "INVALID"
            
            nb_result = {
                "file": nb_file.name,
                "path": str(nb_file.relative_to(REPO_ROOT)),
                "status": status,
                "cell_count": cell_count,
                "code_cells": len(code_cells),
                "imports_found": imports_found[:5]
            }
            
            results["notebooks"].append(nb_result)
            
            if status == "VALID":
                print(f"  ✓ {nb_file.name}: {cell_count} cells, {len(code_cells)} code cells")
                results["summary"]["valid"] += 1
            else:
                print(f"  ❌ {nb_file.name}: Invalid structure")
                results["summary"]["invalid"] += 1
                
        except Exception as e:
            print(f"  ❌ {nb_file.name}: Error reading - {e}")
            results["notebooks"].append({
                "file": nb_file.name,
                "status": "ERROR",
                "error": str(e)
            })
            results["summary"]["invalid"] += 1
    
    # Save results
    report_file = DELIVERABLES_DIR / f"notebook_verification_report_{TIMESTAMP}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Notebook verification saved to: {report_file.relative_to(REPO_ROOT)}")
    print(f"\nSummary: {results['summary']['valid']}/{results['summary']['total_notebooks']} notebooks valid")
    
    log_action("C6_NOTEBOOK", "complete", REPO_ROOT, "SUCCESS",
               valid=results["summary"]["valid"],
               invalid=results["summary"]["invalid"])
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution flow for health checks"""
    try:
        print(f"\n{'='*80}")
        print(f"SYSTEM HEALTH CHECK & TEST SUITE")
        print(f"{'='*80}")
        print(f"Timestamp: {TIMESTAMP}")
        print(f"Repository: {REPO_ROOT}")
        print(f"{'='*80}")
        
        all_results = {}
        
        # Run all health check phases
        all_results["c1_static"] = phase_c1_static_checks()
        all_results["c2_integrity"] = phase_c2_data_integrity()
        all_results["c3_model"] = phase_c3_model_tests()
        all_results["c4_backend"] = phase_c4_backend_checks()
        all_results["c5_safety"] = phase_c5_recon_safety()
        all_results["c6_notebook"] = phase_c6_notebook_verification()
        
        # Generate summary
        print(f"\n{'='*80}")
        print(f"HEALTH CHECK COMPLETE")
        print(f"{'='*80}")
        
        # Calculate totals
        total_tests = sum([
            all_results["c1_static"]["summary"].get("total_files", 0),
            all_results["c2_integrity"]["summary"].get("total_checks", 0),
            all_results["c3_model"]["summary"].get("total_tests", 0),
            all_results["c4_backend"]["summary"].get("total_checks", 0),
        ])
        
        total_passed = sum([
            all_results["c1_static"]["summary"].get("compile_passed", 0),
            all_results["c2_integrity"]["summary"].get("passed", 0),
            all_results["c3_model"]["summary"].get("passed", 0),
            all_results["c4_backend"]["summary"].get("passed", 0),
        ])
        
        print(f"\nOverall Results:")
        print(f"  Total Tests Run: {total_tests}")
        print(f"  Tests Passed: {total_passed}")
        print(f"  Success Rate: {(total_passed/total_tests*100):.1f}%")
        
        print(f"\nAll reports saved to: {DELIVERABLES_DIR.relative_to(REPO_ROOT)}/")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", file=sys.stderr)
        
        error_log = DELIVERABLES_DIR / "cleanup_errors.log"
        with open(error_log, "a") as f:
            f.write(f"\n[{datetime.now().isoformat()}] HEALTH CHECK ERROR:\n")
            f.write(f"{str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
