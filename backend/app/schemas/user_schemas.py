from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

class UserRole:
    USER = "USER"
    ADMIN = "ADMIN"

class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: Optional[str] = None
    role: Optional[str] = UserRole.USER

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class TokenData(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None
