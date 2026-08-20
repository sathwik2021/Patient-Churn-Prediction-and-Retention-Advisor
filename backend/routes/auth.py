"""
Authentication Routes — Patient Churn Prediction
=================================================
Endpoints: /api/auth/signup, /api/auth/signin, /api/auth/signout, /api/auth/me
Uses simple token-based auth with UUID tokens stored in memory.
"""

import uuid
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

import database as db

router = APIRouter(prefix="/api/auth", tags=["authentication"])


class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class UserResponse(BaseModel):
    user: dict


def get_current_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract user_id from Bearer token."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "")
    return db.get_user_id_by_token(token)


@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignUpRequest):
    user = db.create_user(req.name, req.email, req.password)
    if not user:
        raise HTTPException(status_code=400, detail="Email already registered")
    token = str(uuid.uuid4())
    db.create_session(token, user["id"])
    return AuthResponse(token=token, user=user)


@router.post("/signin", response_model=AuthResponse)
async def signin(req: SignInRequest):
    user = db.authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = str(uuid.uuid4())
    db.create_session(token, user["id"])
    return AuthResponse(token=token, user=user)


@router.get("/me", response_model=UserResponse)
async def get_me(authorization: Optional[str] = Header(None)):
    user_id = get_current_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return UserResponse(user=user)


@router.post("/signout")
async def signout(authorization: Optional[str] = Header(None)):
    if authorization:
        db.delete_session(authorization.replace("Bearer ", ""))
    return {"status": "signed_out"}
