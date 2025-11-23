"""
FastAPI Model Server - Rule-based SQL Injection Detection
Provides REST API for inference with regex-based detection
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import re
import json
import os
from datetime import datetime, timezone

# Import ML model wrapper (optional - only if models available)
try:
    from backend.models.Bestmodel import BestModel
    ML_MODEL_AVAILABLE = True
    ml_model = None  # Will be initialized on first use
except ImportError:
    ML_MODEL_AVAILABLE = False
    ml_model = None


app = FastAPI(
    title="SQL Injection Detection API",
    description="Rule-based SQL injection detection system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)


class InferenceRequest(BaseModel):
    """Request model for inference"""
    method: str
    url: str
    params: Optional[Dict] = None
    body: Optional[str] = None
    headers: Optional[Dict] = None
    raw_query: Optional[str] = None
    source_ip: Optional[str] = None


class InferenceResponse(BaseModel):
    """Response model for inference"""
    score: float
    action: str  # "allow", "block", "challenge"
    reason: str
    matched_rules: List[str]
    features: Dict


class DetectionEngine:
    """Rule-based SQL injection detection engine"""
    
    def __init__(self):
        self.rules = self._load_rules()
        
    def _load_rules(self) -> List[Dict]:
        """Load detection rules"""
        return [
            {
                "name": "UNION SELECT",
                "pattern": r"UNION\s+(ALL\s+)?SELECT",
                "flags": re.IGNORECASE,
                "severity": "high",
                "score": 0.95
            },
            {
                "name": "Boolean-based blind",
                "pattern": r"(\bOR\b\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?)|(\bAND\b\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?)",
                "flags": re.IGNORECASE,
                "severity": "high",
                "score": 0.90
            },
            {
                "name": "SQL time-based",
                "pattern": r"(SLEEP\s*\(|BENCHMARK\s*\(|pg_sleep\s*\(|WAITFOR\s+DELAY)",
                "flags": re.IGNORECASE,
                "severity": "high",
                "score": 0.95
            },
            {
                "name": "SQL comments",
                "pattern": r"(--|;--|/\*|\*/|#)",
                "flags": 0,
                "severity": "medium",
                "score": 0.60
            },
            {
                "name": "Information schema access",
                "pattern": r"information_schema\.(tables|columns|schemata)",
                "flags": re.IGNORECASE,
                "severity": "high",
                "score": 0.90
            },
            {
                "name": "System command execution",
                "pattern": r"(xp_cmdshell|sp_configure|exec\s+master|load_file|into\s+outfile|into\s+dumpfile)",
                "flags": re.IGNORECASE,
                "severity": "critical",
                "score": 1.0
            },
            {
                "name": "SQL error-based",
                "pattern": r"(extractvalue|updatexml|exp\(|floor\(rand|row\(|multipoint|polygon|geometrycollection)",
                "flags": re.IGNORECASE,
                "severity": "high",
                "score": 0.85
            },
            {
                "name": "Stacked queries",
                "pattern": r";\s*(DROP|INSERT|UPDATE|DELETE|CREATE|ALTER)\s+",
                "flags": re.IGNORECASE,
                "severity": "critical",
                "score": 1.0
            },
            {
                "name": "Database functions",
                "pattern": r"(database\(\)|version\(\)|user\(\)|@@version|current_user)",
                "flags": re.IGNORECASE,
                "severity": "medium",
                "score": 0.70
            },
            {
                "name": "Hex encoding",
                "pattern": r"0x[0-9a-fA-F]{10,}",
                "flags": 0,
                "severity": "medium",
                "score": 0.65
            },
            {
                "name": "CHAR/ASCII manipulation",
                "pattern": r"(CHAR\s*\(|ASCII\s*\(|CONCAT\s*\(|SUBSTRING\s*\(|MID\s*\()",
                "flags": re.IGNORECASE,
                "severity": "medium",
                "score": 0.60
            },
            {
                "name": "Conditional statements",
                "pattern": r"(IF\s*\(|CASE\s+WHEN|IIF\s*\()",
                "flags": re.IGNORECASE,
                "severity": "medium",
                "score": 0.55
            },
            {
                "name": "Multiple SQL keywords",
                "pattern": r"(SELECT\s+.*\s+FROM\s+.*\s+(WHERE|UNION|AND|OR)\s+['\"]?\d+['\"]?\s*=)",
                "flags": re.IGNORECASE,
                "severity": "medium",
                "score": 0.75
            },
            {
                "name": "Quote escaping attempts",
                "pattern": r"(\\['\"\\]|%27|%22|%5C)",
                "flags": 0,
                "severity": "medium",
                "score": 0.50
            },
            {
                "name": "Blind injection inference",
                "pattern": r"(SUBSTRING|SUBSTR|MID|ASCII|ORD)\s*\([^)]*\)\s*[><=]",
                "flags": re.IGNORECASE,
                "severity": "high",
                "score": 0.85
            },
        ]
    
    def detect(self, text: str) -> tuple:
        """
        Detect SQL injection patterns
        Returns: (max_score, matched_rules)
        """
        if not text:
            return 0.0, []
        
        matched_rules = []
        max_score = 0.0
        
        for rule in self.rules:
            pattern = re.compile(rule["pattern"], rule["flags"])
            if pattern.search(text):
                matched_rules.append(rule["name"])
                max_score = max(max_score, rule["score"])
        
        return max_score, matched_rules
    
    def extract_features(self, request: InferenceRequest) -> Dict:
        """Extract features from request"""
        raw = request.raw_query or ""
        
        if not raw:
            # Build raw from components
            parts = [request.url or ""]
            if request.params:
                parts.append(str(request.params))
            if request.body:
                parts.append(request.body)
            if request.headers:
                parts.append(str(request.headers))
            raw = " ".join(parts)
        
        features = {
            "len_raw": len(raw),
            "count_susp_chars": sum([
                raw.count("'"),
                raw.count('"'),
                raw.count("--"),
                raw.count(";"),
                raw.count("/*"),
                raw.count("*/"),
            ]),
            "num_sql_keywords": sum([
                1 for kw in ["SELECT", "UNION", "INSERT", "UPDATE", "DELETE", 
                            "DROP", "CREATE", "ALTER", "WHERE", "FROM"]
                if kw in raw.upper()
            ]),
            "has_union": "UNION" in raw.upper(),
            "has_or_equals": bool(re.search(r"OR\s+['\"]?1['\"]?\s*=", raw, re.I)),
            "has_sleep": bool(re.search(r"SLEEP\s*\(", raw, re.I)),
            "has_comments": bool(re.search(r"(--|/\*|\*/)", raw)),
            "url_encoded": bool(re.search(r"%[0-9a-fA-F]{2}", raw)),
            "base64_like": bool(re.search(r"[A-Za-z0-9+/]{20,}={0,2}", raw)),
        }
        
        return features
    
    def infer(self, request: InferenceRequest, enable_challenge: bool = False) -> InferenceResponse:
        """
        Perform inference on request
        """
        # Build text to analyze
        text_parts = []
        
        if request.url:
            text_parts.append(request.url)
        
        if request.params:
            text_parts.append(json.dumps(request.params))
        
        if request.body:
            text_parts.append(request.body)
        
        if request.headers:
            text_parts.append(json.dumps(request.headers))
        
        if request.raw_query:
            text_parts.append(request.raw_query)
        
        text = " ".join(text_parts)
        
        # Detect patterns
        score, matched_rules = self.detect(text)
        
        # Extract features
        features = self.extract_features(request)
        
        # Determine action
        if score >= 0.80:
            action = "block"
            reason = "rule"
        elif enable_challenge and 0.50 <= score < 0.80:
            action = "challenge"
            reason = "heuristic"
        else:
            action = "allow"
            reason = "none"
        
        return InferenceResponse(
            score=score,
            action=action,
            reason=reason,
            matched_rules=matched_rules,
            features=features
        )


# Global detection engine
engine = DetectionEngine()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "sql-injection-detection",
        "version": "1.0.0"
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Perform inference on a request
    """
    try:
        # Perform inference
        result = engine.infer(request)
        
        # Log request and response
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": request.source_ip,
            "method": request.method,
            "url": request.url,
            "score": result.score,
            "action": result.action,
            "matched_rules": result.matched_rules,
            "features": result.features,
        }
        
        with open("logs/requests.jl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ML MODEL ENDPOINT - XGBoost-based detection
# ============================================================================

class MLInferenceRequest(BaseModel):
    """Request model for ML inference"""
    raw_query: str
    threshold_mode: Optional[str] = "balanced"  # balanced, high_security, high_availability


class MLInferenceResponse(BaseModel):
    """Response model for ML inference"""
    score: float
    action: str  # "allow", "challenge", "block"
    reason: str
    features: Dict
    confidence: str
    threshold_mode: str
    model_version: str


@app.post("/infer-ml", response_model=MLInferenceResponse)
async def infer_ml(request: MLInferenceRequest):
    """
    Perform ML-based inference using trained XGBoost model
    
    This endpoint uses the production XGBoost model for SQL injection detection.
    It provides more sophisticated analysis compared to the rule-based /infer endpoint.
    
    Args:
        request: MLInferenceRequest with raw_query and optional threshold_mode
        
    Returns:
        MLInferenceResponse with score, action, reason, and extracted features
        
    Raises:
        HTTPException: If ML model is not available or inference fails
    """
    global ml_model
    
    try:
        # Check if ML model is available
        if not ML_MODEL_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="ML model not available. Please ensure model artifacts are in backend/models/"
            )
        
        # Lazy initialization of ML model
        if ml_model is None:
            try:
                ml_model = BestModel(threshold_mode=request.threshold_mode)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load ML model: {str(e)}"
                )
        
        # Perform prediction
        result = ml_model.predict(request.raw_query)
        
        # Check for prediction errors
        if result.get("action") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.get('reason', 'Unknown error')}"
            )
        
        # Log request and response
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": "/infer-ml",
            "raw_query": request.raw_query[:200],  # Truncate for logs
            "score": result["score"],
            "action": result["action"],
            "confidence": result["confidence"],
            "threshold_mode": result["threshold_mode"],
        }
        
        with open("logs/requests.jl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Return response
        return MLInferenceResponse(
            score=result["score"],
            action=result["action"],
            reason=result["reason"],
            features=result["features"],
            confidence=result["confidence"],
            threshold_mode=result["threshold_mode"],
            model_version=result["model_version"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/ml-model-info")
async def get_ml_model_info():
    """
    Get information about the loaded ML model
    
    Returns model metadata, metrics, and configuration
    """
    global ml_model
    
    if not ML_MODEL_AVAILABLE:
        return {
            "available": False,
            "message": "ML model not available"
        }
    
    try:
        # Initialize model if needed
        if ml_model is None:
            ml_model = BestModel()
        
        info = ml_model.get_model_info()
        return {
            "available": True,
            **info
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


# ============================================================================
# END ML MODEL ENDPOINTS
# ============================================================================
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get statistics from logs"""
    try:
        if not os.path.exists("logs/requests.jl"):
            return {
                "total_requests": 0,
                "blocked": 0,
                "allowed": 0,
                "challenged": 0
            }
        
        stats = {
            "total_requests": 0,
            "blocked": 0,
            "allowed": 0,
            "challenged": 0,
            "avg_score": 0.0,
        }
        
        scores = []
        
        with open("logs/requests.jl", "r") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                    stats["total_requests"] += 1
                    
                    action = entry.get("action", "allow")
                    stats[action] = stats.get(action, 0) + 1
                    
                    scores.append(entry.get("score", 0.0))
                except:
                    continue
        
        if scores:
            stats["avg_score"] = sum(scores) / len(scores)
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rules")
async def get_rules():
    """Get detection rules"""
    return {
        "total_rules": len(engine.rules),
        "rules": [
            {
                "name": rule["name"],
                "severity": rule["severity"],
                "score": rule["score"]
            }
            for rule in engine.rules
        ]
    }


@app.post("/batch")
async def batch_infer(payload: Dict):
    """
    Perform batch inference on multiple requests
    """
    try:
        requests_data = payload.get("requests", [])
        results = []
        
        for req_data in requests_data:
            # Convert dict to InferenceRequest
            request = InferenceRequest(**req_data)
            result = engine.infer(request)
            results.append({
                "score": result.score,
                "action": result.action,
                "reason": result.reason,
                "matched_rules": result.matched_rules
            })
        
        return {
            "total": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
