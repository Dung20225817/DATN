from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey
from sqlalchemy.sql import func
from app.db_connect import Base

class User(Base):
    __tablename__ = "users"

    uuid = Column(Integer, primary_key=True, index=True)
    user_name = Column(String)
    email = Column(String)
    phone = Column(String)
    password = Column(String)

class Picture(Base):
    __tablename__ = "pictures"

    pid = Column(Integer, primary_key=True, index=True)
    p_name = Column(String)
    save_time = Column(TIMESTAMP, server_default=func.now())
    uuid = Column(Integer, ForeignKey("users.uuid", ondelete="CASCADE"))
