from sqlalchemy import Column, Integer, String
from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    uuid = Column(Integer, primary_key=True, index=True)
    user_name = Column(String)
    email = Column(String)
    phone = Column(String)
    password = Column(String)
