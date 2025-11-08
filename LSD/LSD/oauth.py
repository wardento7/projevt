from jose import JWTError, jwt
from datetime import datetime, timedelta
from LSD import schema, model
from fastapi import Depends, status, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from LSD.database import get_db
import io, base64
import matplotlib.pyplot as plt
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
SECRET_KEY = "ad46747c61b19640be0754cf30f53dfc95cf9b5d4673eb77e92a9f67a60e6b37"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
def verify_token_access(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        return schema.TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    if not token:
        token = request.cookies.get("access_token")
    if not token:
        raise credentials_exception
    token_data = verify_token_access(token, credentials_exception)
    user = db.query(model.User).filter(model.User.id == token_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
