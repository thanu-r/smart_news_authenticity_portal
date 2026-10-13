from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, model_validator

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
    username_or_email: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    password: str

    @model_validator(mode="before")
    @classmethod
    def resolve_login_field(cls, values):
        if isinstance(values, dict):
            ident = values.get("username_or_email") or values.get("email") or values.get("username")
            if ident:
                values["username_or_email"] = ident
        return values

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
