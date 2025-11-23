#!/usr/bin/env python3
"""
مثال توضيحي: كيفية الربط مع API الباك اند
Integration Example: How to connect with the Backend API
"""

import requests
import json
from typing import Dict, Any


# ═══════════════════════════════════════════════════════════════════════════
# الإعدادات الأساسية / Basic Configuration
# ═══════════════════════════════════════════════════════════════════════════

API_BASE_URL = "http://localhost:8000"  # عنوان السيرفر


# ═══════════════════════════════════════════════════════════════════════════
# دالة للربط مع الباك اند / Function to connect with backend
# ═══════════════════════════════════════════════════════════════════════════

def check_sql_injection(method: str, url: str, params: Dict = None, 
                       body: str = None, headers: Dict = None,
                       use_ml: bool = True) -> Dict[str, Any]:
    """
    فحص الطلب للكشف عن SQL Injection
    Check request for SQL Injection detection
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        params: Query parameters (optional)
        body: Request body (optional)
        headers: Request headers (optional)
        use_ml: Use ML model (True) or rule-based (False)
    
    Returns:
        Dictionary with detection results
    """
    # اختيار الـ endpoint / Choose endpoint
    endpoint = "/infer-ml" if use_ml else "/infer"
    
    # تحضير البيانات / Prepare data
    data = {
        "method": method,
        "url": url
    }
    
    if params:
        data["params"] = params
    if body:
        data["body"] = body
    if headers:
        data["headers"] = headers
    
    try:
        # إرسال الطلب / Send request
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=data,
            timeout=10
        )
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.ConnectionError:
        return {
            "error": "Cannot connect to backend server",
            "message": "تأكد من تشغيل السيرفر: cd backend && uvicorn model_server:app"
        }
    except Exception as e:
        return {
            "error": str(e)
        }


def get_model_info() -> Dict[str, Any]:
    """
    الحصول على معلومات الموديل
    Get model information
    """
    try:
        response = requests.get(f"{API_BASE_URL}/ml-model-info", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def check_health() -> Dict[str, Any]:
    """
    فحص صحة السيرفر
    Check server health
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# أمثلة على الاستخدام / Usage Examples
# ═══════════════════════════════════════════════════════════════════════════

def example_1_simple_query():
    """مثال 1: فحص استعلام بسيط"""
    print("\n" + "="*70)
    print("مثال 1: فحص استعلام بسيط / Example 1: Simple Query Check")
    print("="*70)
    
    result = check_sql_injection(
        method="GET",
        url="/users?id=1",
        params={"id": "1"}
    )
    
    print(f"النتيجة / Result: {result['action']}")
    print(f"الدرجة / Score: {result['score']}")
    print(f"السبب / Reason: {result['reason']}")


def example_2_sql_injection():
    """مثال 2: كشف SQL Injection"""
    print("\n" + "="*70)
    print("مثال 2: كشف SQL Injection / Example 2: SQL Injection Detection")
    print("="*70)
    
    result = check_sql_injection(
        method="GET",
        url="/users?id=1 OR 1=1--",
        params={"id": "1 OR 1=1--"}
    )
    
    print(f"النتيجة / Result: {result['action']}")
    print(f"الدرجة / Score: {result['score']}")
    print(f"السبب / Reason: {result['reason']}")
    print(f"الثقة / Confidence: {result['confidence']}")


def example_3_post_request():
    """مثال 3: فحص POST request مع body"""
    print("\n" + "="*70)
    print("مثال 3: فحص POST Request / Example 3: POST Request Check")
    print("="*70)
    
    result = check_sql_injection(
        method="POST",
        url="/login",
        body="username=admin&password=' OR '1'='1",
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    print(f"النتيجة / Result: {result['action']}")
    print(f"الدرجة / Score: {result['score']}")
    print(f"السبب / Reason: {result['reason']}")


def example_4_model_info():
    """مثال 4: الحصول على معلومات الموديل"""
    print("\n" + "="*70)
    print("مثال 4: معلومات الموديل / Example 4: Model Information")
    print("="*70)
    
    info = get_model_info()
    
    if "error" not in info:
        print(f"نوع الموديل / Model Type: {info['model_type']}")
        print(f"الإصدار / Version: {info['version']}")
        print(f"F1 Score: {info['metrics']['f1']:.4f}")
        print(f"Precision: {info['metrics']['precision']:.4f}")
        print(f"Recall: {info['metrics']['recall']:.4f}")
    else:
        print(f"خطأ / Error: {info['error']}")


def example_5_health_check():
    """مثال 5: فحص صحة السيرفر"""
    print("\n" + "="*70)
    print("مثال 5: فحص صحة السيرفر / Example 5: Health Check")
    print("="*70)
    
    health = check_health()
    
    if "error" not in health:
        print(f"الحالة / Status: {health['status']}")
        print(f"ML Model Available: {health.get('ml_model_available', False)}")
    else:
        print(f"خطأ / Error: {health['error']}")


def example_6_batch_checking():
    """مثال 6: فحص مجموعة من الطلبات"""
    print("\n" + "="*70)
    print("مثال 6: فحص دفعة / Example 6: Batch Checking")
    print("="*70)
    
    test_queries = [
        ("GET", "/search?q=python", {"q": "python"}),
        ("GET", "/users?id=1", {"id": "1"}),
        ("GET", "/admin?id=1' OR '1'='1", {"id": "1' OR '1'='1"}),
        ("GET", "/data?id=1 UNION SELECT * FROM users--", {"id": "1 UNION SELECT * FROM users--"}),
    ]
    
    results = []
    for method, url, params in test_queries:
        result = check_sql_injection(method, url, params)
        results.append({
            "url": url,
            "action": result.get("action", "error"),
            "score": result.get("score", -1)
        })
    
    print("\nالنتائج / Results:")
    print(f"{'URL':<50} {'Action':<10} {'Score':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['url']:<50} {r['action']:<10} {r['score']:<10.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# دالة middleware للتطبيقات الكبيرة / Middleware function for larger apps
# ═══════════════════════════════════════════════════════════════════════════

class SQLInjectionMiddleware:
    """
    Middleware class للاستخدام مع Flask أو Django
    Middleware for use with Flask or Django
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
    
    def check_request(self, request) -> Dict[str, Any]:
        """
        فحص الطلب الوارد
        Check incoming request
        
        Args:
            request: Flask/Django request object
        
        Returns:
            Detection result dictionary
        """
        # استخراج البيانات من الطلب / Extract data from request
        method = request.method
        url = request.path
        params = dict(request.args) if hasattr(request, 'args') else {}
        body = request.get_data(as_text=True) if hasattr(request, 'get_data') else None
        headers = dict(request.headers) if hasattr(request, 'headers') else {}
        
        # فحص الطلب / Check request
        return check_sql_injection(method, url, params, body, headers)
    
    def should_block(self, request) -> bool:
        """
        هل يجب حظر الطلب؟
        Should the request be blocked?
        
        Returns:
            True if request should be blocked
        """
        result = self.check_request(request)
        return result.get("action") == "block"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("مثال تكامل API الباك اند / Backend API Integration Example")
    print("="*70)
    print("\nملاحظة: تأكد من تشغيل السيرفر أولاً:")
    print("Note: Make sure to start the server first:")
    print("  cd backend && uvicorn model_server:app --host 0.0.0.0 --port 8000")
    print("="*70)
    
    # تشغيل الأمثلة / Run examples
    try:
        example_5_health_check()
        example_4_model_info()
        example_1_simple_query()
        example_2_sql_injection()
        example_3_post_request()
        example_6_batch_checking()
        
        print("\n" + "="*70)
        print("✅ جميع الأمثلة انتهت بنجاح / All examples completed successfully")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ خطأ / Error: {e}")
        print("\nتأكد من:")
        print("1. تشغيل السيرفر (cd backend && uvicorn model_server:app)")
        print("2. تثبيت المكتبات (pip install requests)")
