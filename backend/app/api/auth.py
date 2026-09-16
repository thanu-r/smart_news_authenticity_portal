from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId
from backend.app.database.connection import get_db
from backend.app.schemas.user_schemas import UserRegister, UserLogin, UserResponse, TokenResponse, UserRole
from backend.app.auth.security import hash_password, verify_password, create_access_token
from backend.app.auth.jwt_handler import get_current_active_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserRegister):
    """Register a new user, returning JWT access token and profile info."""
    db = get_db()
    
    # Check if email already registered
    existing_email = await db.users.find_one({"email": user_in.email.lower()})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered"
        )
        
    # Check if username already exists
    existing_username = await db.users.find_one({"username": user_in.username})
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already taken"
        )
        
    # Check if this is the very first user; if so or if explicitly requested as ADMIN, grant ADMIN
    total_users = await db.users.count_documents({})
    assigned_role = UserRole.ADMIN if total_users == 0 else (user_in.role or UserRole.USER)
    if assigned_role not in [UserRole.USER, UserRole.ADMIN]:
        assigned_role = UserRole.USER

    now = datetime.now(timezone.utc)
    hashed_pw = hash_password(user_in.password)
    
    user_doc = {
        "email": user_in.email.lower(),
        "username": user_in.username,
        "hashed_password": hashed_pw,
        "full_name": user_in.full_name or "",
        "role": assigned_role,
        "is_active": True,
        "created_at": now,
        "updated_at": now
    }
    
    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    
    access_token = create_access_token(data={"sub": user_id, "role": assigned_role})
    
    user_resp = UserResponse(
        id=user_id,
        email=user_doc["email"],
        username=user_doc["username"],
        full_name=user_doc["full_name"],
        role=assigned_role,
        is_active=True,
        created_at=now
    )
    
    return TokenResponse(access_token=access_token, token_type="bearer", user=user_resp)

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """Authenticate user with username or email and return JWT token."""
    db = get_db()
    login_id = credentials.username_or_email.strip().lower()
    
    user_doc = await db.users.find_one({
        "$or": [
            {"email": login_id},
            {"username": credentials.username_or_email.strip()}
        ]
    })
    
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
        
    if not verify_password(credentials.password, user_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
        
    if not user_doc.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is disabled"
        )
        
    user_id = str(user_doc["_id"])
    role = user_doc.get("role", UserRole.USER)
    token = create_access_token(data={"sub": user_id, "role": role})
    
    user_resp = UserResponse(
        id=user_id,
        email=user_doc["email"],
        username=user_doc["username"],
        full_name=user_doc.get("full_name"),
        role=role,
        is_active=True,
        created_at=user_doc["created_at"]
    )
    
    return TokenResponse(access_token=token, token_type="bearer", user=user_resp)

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_active_user)):
    """Return currently authenticated user profile."""
    return current_user
