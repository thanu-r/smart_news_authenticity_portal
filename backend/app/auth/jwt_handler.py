from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
from backend.app.auth.security import decode_access_token
from backend.app.database.connection import get_db
from backend.app.schemas.user_schemas import UserResponse, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> UserResponse:
    """Dependency that extracts and validates the authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    db = get_db()
    user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
    if user_doc is None:
        raise credentials_exception

    return UserResponse(
        id=str(user_doc["_id"]),
        email=user_doc["email"],
        username=user_doc["username"],
        full_name=user_doc.get("full_name"),
        role=user_doc.get("role", UserRole.USER),
        is_active=user_doc.get("is_active", True),
        created_at=user_doc["created_at"],
    )

async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Ensure the user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user account")
    return current_user

async def get_admin_user(current_user: UserResponse = Depends(get_current_active_user)) -> UserResponse:
    """Ensure the user has ADMIN role privileges."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation"
        )
    return current_user

async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[UserResponse]:
    """Allow optional authentication for public endpoints."""
    if not token:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None
