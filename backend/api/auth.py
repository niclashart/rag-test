"""Authentication endpoints."""
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import os
from dotenv import load_dotenv

from database.database import get_db
from database.crud import create_user, get_user_by_email
from backend.dependencies import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_active_user
)
from database.models import User
from logging_config.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

router = APIRouter(prefix="/api/auth", tags=["auth"])


class UserRegister(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    created_at: Optional[str] = None

    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, obj: User):
        """Convert User ORM object to UserResponse."""
        return cls(
            id=obj.id,
            email=obj.email,
            is_active=obj.is_active,
            created_at=obj.created_at.isoformat() if obj.created_at else None
        )


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        logger.info(f"Registration attempt for email: {user_data.email}")
        
        # Check if user already exists
        db_user = get_user_by_email(db, email=user_data.email)
        if db_user:
            logger.warning(f"Registration failed: Email {user_data.email} already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        user = create_user(db, email=user_data.email, hashed_password=hashed_password)
        logger.info(f"User created successfully: {user.id}")
        
        # Convert to response model
        return UserResponse.from_orm(user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token."""
    user = get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return UserResponse.from_orm(current_user)


