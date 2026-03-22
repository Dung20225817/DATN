# be/api/auth.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db_connect import get_db
from app.db.table import User

router = APIRouter()

class LoginData(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(data: LoginData, db: Session = Depends(get_db)):

    # Tìm user theo email
    user = db.query(User).filter(User.email == data.email).first()

    if user is None:
        raise HTTPException(status_code=401, detail="Email không tồn tại")

    # Kiểm tra password (tạm thời là plain text)
    if user.password != data.password:
        raise HTTPException(status_code=401, detail="Sai mật khẩu")

    # Nếu đúng → trả về thông tin user
    return {
        "uid": user.uuid,
        "email": user.email,
        "user_name": user.user_name,
        "phone": user.phone,
        "token": "fake_jwt_123"  # sau này có thể thay bằng JWT thật
    }