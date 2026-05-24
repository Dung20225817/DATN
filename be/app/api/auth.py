import os

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.db.session import get_db

router = APIRouter()
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


class LoginData(BaseModel):
    email: str
    password: str


class RegisterData(BaseModel):
    user_name: str
    email: str
    phone: str
    password: str


class GoogleAuthData(BaseModel):
    credential: str


def _serialize_user(user: User) -> dict:
    return {
        "uid": user.uuid,
        "email": user.email,
        "user_name": user.user_name,
        "phone": user.phone,
        "token": "fake_jwt_123",  # later: replace with a real JWT
    }


def _verify_google_credential(credential: str) -> dict:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(status_code=503, detail="Google login chua duoc cau hinh")

    try:
        response = requests.get(
            GOOGLE_TOKENINFO_URL,
            params={"id_token": credential},
            timeout=8,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Khong the xac minh Google token") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Google token khong hop le")

    try:
        token_info = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Phan hoi Google khong hop le") from exc
    if token_info.get("aud") != client_id:
        raise HTTPException(status_code=401, detail="Google token khong dung ung dung")

    email = str(token_info.get("email") or "").strip().lower()
    email_verified = str(token_info.get("email_verified") or "").lower() == "true"
    if not email or not email_verified:
        raise HTTPException(status_code=401, detail="Email Google chua duoc xac minh")

    return token_info


@router.post("/login")
def login(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if user is None:
        raise HTTPException(status_code=401, detail="Email không tồn tại")

    # Passwords are still stored as plain text to preserve current behavior.
    # This should be replaced with proper password hashing in a later security pass.
    if user.password != data.password:
        raise HTTPException(status_code=401, detail="Sai mật khẩu")

    return _serialize_user(user)


@router.post("/register")
def register(data: RegisterData, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email đã tồn tại")

    user = User(
        user_name=data.user_name,
        email=data.email,
        phone=data.phone,
        password=data.password,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _serialize_user(user)


@router.post("/auth/google")
def google_auth(data: GoogleAuthData, db: Session = Depends(get_db)):
    token_info = _verify_google_credential(data.credential)
    email = str(token_info.get("email") or "").strip().lower()

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user_name = str(token_info.get("name") or "").strip() or email.split("@", 1)[0]
        user = User(
            user_name=user_name,
            email=email,
            phone="",
            password="",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return _serialize_user(user)
