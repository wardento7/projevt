from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime
from typing import List
class UserCreate(BaseModel):
    user_name: str
    email:EmailStr
    password:str
    company_name:str
    class Config:
        from_attributes = True
class LoginRequest(BaseModel):
    username: str
    password: str
class UserEmail(BaseModel):
    email: str
    class Config:
        from_attributes = True
class ResetPassword(BaseModel):
    email: str
    otp: str
    new_password: str
    class Config:
        from_attributes = True
class password(BaseModel):
    new_password:str
    confirm_password:str
class TokenData(BaseModel):
    user_id: int
class UsernameChange(BaseModel):
    new_username: str
class EmailChange(BaseModel):
    new_email: EmailStr
class ChatRequest(BaseModel):
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    user_message: str
class ChatResponse(BaseModel):
    session_id: Optional[str]
    user_message: str
    bot_reply: str
class QuestionRequest(BaseModel):
    question: str
class ChatResponse(BaseModel):
    question: str
    answer: str
class ReportResponse(BaseModel):
    id: int
    report_summary: Optional[str] = None
    recommendations: Optional[str] = None
    scan_id: Optional[int] = None
    user_id: int
    generated_at: datetime
    chart_url: Optional[str] = None
    data_json: Optional[Dict[str, Any]] = None
    class Config:
        from_attributes = True
class InferenceRequest(BaseModel):
    method: str
    url: str
    params: Optional[Dict] = None
    body: Optional[str] = None
    headers: Optional[Dict] = None
    raw_query: Optional[str] = None
    source_ip: Optional[str] = None
class InferenceResponse(BaseModel):
    score: float
    action: str 
    reason: str
    matched_rules: List[str]
    features: Dict
