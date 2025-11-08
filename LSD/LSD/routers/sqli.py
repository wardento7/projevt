from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Dict
import json, os
from LSD.database import get_db
from LSD import model, utils, schema, oauth

router = APIRouter(prefix="/scan", tags=["scan"])
engine = utils.DetectionEngine()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "sql-injection-detection",
        "version": "1.0.0"
    }

@router.post("/infer", response_model=schema.InferenceResponse)
async def infer(
    req_data: schema.InferenceRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="❌ Access token not found")
        
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        current_user = oauth.verify_token_access(token, credentials_exception)
        
        result = engine.infer(req_data)
        
        scan_log = model.ScanLog(
            timestamp=datetime.now(timezone.utc),
            source_ip=req_data.source_ip,
            user_id=current_user.user_id,  # ✅ تصحيح من user_id إلى id
            method=req_data.method,
            url=req_data.url,
            score=result.score,
            action=result.action,
            reason=result.reason,
            matched_rules=result.matched_rules,
            features=result.features,
        )
        db.add(scan_log)
        db.commit()
        db.refresh(scan_log)
        
        result_entry = model.Result(
            scan_id=scan_log.id,
            score=result.score,
            action=result.action,
            reason=result.reason,
            matched_rules=result.matched_rules
        )
        db.add(result_entry)
        db.commit()
        
        return result
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/stats")
async def get_stats():
    try:
        if not os.path.exists("logs/requests.jl"):
            return {"total_requests": 0, "blocked": 0, "allowed": 0, "challenged": 0}
        stats = {"total_requests": 0, "blocked": 0, "allowed": 0, "challenged": 0, "avg_score": 0.0}
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

@router.get("/rules")
async def get_rules():
    return {
        "total_rules": len(engine.rules),
        "rules": [
            {"name": rule["name"], "severity": rule["severity"], "score": rule["score"]}
            for rule in engine.rules
        ]
    }

@router.post("/batch")
async def batch_infer(
    payload: Dict,
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="❌ Access token not found")

        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        current_user = oauth.verify_token_access(token, credentials_exception)
        
        requests_data = payload.get("requests", [])
        results = []
        db_entries = []
        
        for req_data in requests_data:
            data = schema.InferenceRequest(**req_data)
            result = engine.infer(data)
            
            db_entry = model.ScanLog(
                timestamp=datetime.now(timezone.utc),
                source_ip=data.source_ip,
                user_id=current_user.user_id,  # ✅ تصحيح من user_id إلى id
                method=data.method,
                url=data.url,
                score=result.score,
                action=result.action,
                reason=result.reason,
                matched_rules=result.matched_rules,
                features=result.features,
            )
            db.add(db_entry)
            db_entries.append((db_entry, result))
        
        db.commit()
        
        for db_entry, result in db_entries:
            db.refresh(db_entry)
            results.append({
                "id": db_entry.id,
                "score": result.score,
                "action": result.action,
                "reason": result.reason,
                "matched_rules": result.matched_rules
            })
        
        return {"total": len(results), "results": results}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
