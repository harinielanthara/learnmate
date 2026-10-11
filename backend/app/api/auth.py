import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..config import get_settings
from ..db import get_db
from ..models import User
from ..schemas import Token, UserCreate, UserOut
from ..security import create_access_token, hash_password, verify_password
from .deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(body: UserCreate, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    if len(body.password.encode("utf-8")) > 72:
        raise HTTPException(400, "Password is too long (max 72 bytes).")
    if db.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(409, "Email already registered.")
    user = User(email=email, password_hash=hash_password(body.password),
                display_name=(body.display_name or email.split("@")[0]).strip()[:100])
    db.add(user)
    db.commit()
    return user


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Form fields are named username/password (OAuth2 standard); username = email."""
    user = db.scalar(select(User).where(User.email == form.username.strip().lower()))
    if not user or not user.password_hash or not verify_password(form.password, user.password_hash):
        # same message for both cases: don't reveal which emails exist
        raise HTTPException(401, "Incorrect email or password.", headers={"WWW-Authenticate": "Bearer"})
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.delete("/me", status_code=204)
def delete_account(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Right-to-delete: removes uploaded files; DB rows go via ON DELETE CASCADE."""
    shutil.rmtree(Path(get_settings().storage_dir) / str(user.id), ignore_errors=True)
    db.delete(user)
    db.commit()
