#!/usr/bin/env python3
"""
Test script to verify all installed packages
"""

import sys
import importlib
from datetime import datetime

def test_import(package_name, display_name=None):
    """Test if a package can be imported"""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name:25s} version: {version}")
        return True
    except ImportError as e:
        print(f"❌ {display_name:25s} FAILED: {str(e)}")
        return False

def main():
    print("=" * 80)
    print("🔍 Testing All Installed Packages")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"Test date: {datetime.now().isoformat()}")
    print("=" * 80)
    print()
    
    results = {}
    
    # Backend Framework
    print("📦 Backend Framework:")
    print("-" * 80)
    results['fastapi'] = test_import('fastapi', 'FastAPI')
    results['uvicorn'] = test_import('uvicorn', 'Uvicorn')
    results['pydantic'] = test_import('pydantic', 'Pydantic')
    results['starlette'] = test_import('starlette', 'Starlette')
    print()
    
    # Data Processing
    print("📊 Data Processing:")
    print("-" * 80)
    results['pandas'] = test_import('pandas', 'Pandas')
    results['numpy'] = test_import('numpy', 'NumPy')
    results['pyarrow'] = test_import('pyarrow', 'PyArrow')
    print()
    
    # Testing
    print("🧪 Testing:")
    print("-" * 80)
    results['pytest'] = test_import('pytest', 'Pytest')
    results['httpx'] = test_import('httpx', 'HTTPX')
    print()
    
    # Security
    print("🔐 Security:")
    print("-" * 80)
    results['jose'] = test_import('jose', 'python-jose')
    results['passlib'] = test_import('passlib', 'Passlib')
    results['cryptography'] = test_import('cryptography', 'Cryptography')
    results['bcrypt'] = test_import('bcrypt', 'Bcrypt')
    print()
    
    # Machine Learning
    print("🤖 Machine Learning:")
    print("-" * 80)
    results['sklearn'] = test_import('sklearn', 'scikit-learn')
    results['xgboost'] = test_import('xgboost', 'XGBoost')
    results['scipy'] = test_import('scipy', 'SciPy')
    print()
    
    # Utilities
    print("🛠️  Utilities:")
    print("-" * 80)
    results['requests'] = test_import('requests', 'Requests')
    results['click'] = test_import('click', 'Click')
    results['yaml'] = test_import('yaml', 'PyYAML')
    print()
    
    # Summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total packages tested: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print("=" * 80)
    
    if failed > 0:
        print("\n⚠️  Some packages failed to import. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All packages imported successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
