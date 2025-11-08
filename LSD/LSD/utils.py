from passlib.context import CryptContext
from typing import  Dict, List
import re
from LSD import schema
import json
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash(password: str) -> str:
    return pwd_context.hash(password)
def verify(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

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
    
    def extract_features(self, request: schema.InferenceRequest) -> Dict:
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
    
    def infer(self, request: schema.InferenceRequest, enable_challenge: bool = False) -> schema.InferenceResponse:
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
        
        return schema.InferenceResponse(
            score=score,
            action=action,
            reason=reason,
            matched_rules=matched_rules,
            features=features
        )
