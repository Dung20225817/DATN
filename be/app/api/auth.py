from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.db.session import get_db

router = APIRouter()


class LoginData(BaseModel):
    email: str
    password: str


class RegisterData(BaseModel):
    user_name: str
    email: str
    phone: str
    password: str


def _serialize_user(user: User) -> dict:
    return {
        "uid": user.uuid,
        "email": user.email,
        "user_name": user.user_name,
        "phone": user.phone,
        "token": "fake_jwt_123",  # later: replace with a real JWT
    }


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
