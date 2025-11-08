from fastapi import APIRouter, Depends, HTTPException, Header, status, Request
from sqlalchemy.orm import Session
from LSD.database import get_db
from LSD import oauth, model, schema
router = APIRouter(prefix="/report", tags=["Report"])
@router.get("/all", response_model=list[schema.ReportResponse])
async def get_my_reports(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing"
        )
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    current_user = oauth.verify_token_access(token, credentials_exception)
    reports = db.query(model.Report).filter(model.Report.user_id == current_user.user_id).all()
    if not reports:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No reports found for this user")
    return reports
@router.get("/search/{report_id}", response_model=schema.ReportResponse)
async def search_report_by_id(report_id: int, request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Access token is missing")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    current_user = oauth.verify_token_access(token, credentials_exception)
    report = (
        db.query(model.Report)
        .join(model.Scan)
        .filter(model.Scan.user_id == current_user.user_id, model.Report.id == report_id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="⚠️ Report not found")
    scan = report.scan 
    return schema.ReportResponse(
        scan_id=report.scan_id,
        target_url=report.target_url,
        sqli_score=report.sqli_score,
        vulnerability_type=report.vulnerability_type,
        findings=report.findings,
        ai_analysis=report.ai_analysis,
        report_summary=report.report_summary,
        chart=report.chart,
        created_at=report.created_at,
    )
@router.delete("/confirm-delete/{report_id}")
async def delete_report(report_id: int, request: Request, db: Session = Depends(get_db)):
    try:
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="❌ Access token is required")
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="❌ Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        current_user = oauth.verify_token_access(token, credentials_exception)
        report = (
            db.query(model.Report)
            .join(model.Scan)
            .filter(
                model.Scan.user_id == current_user.user_id,
                model.Report.id == report_id
            )
            .first()
        )
        if not report:
            raise HTTPException(status_code=404, detail="⚠️ Report not found or not yours")
        db.delete(report)
        db.commit()
        return {
            "message": "✅ Report deleted successfully",
            "report_id": report_id
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
