# app/db/ocr_tables.py

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from datetime import datetime

from app.db_connect import Base


class OMRTest(Base):
    """Luu thong tin de OMR do giao vien tao."""

    __tablename__ = "omr_test"

    omrid = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(Integer, ForeignKey("users.uuid"), nullable=False)
    omr_name = Column(String, nullable=False)
    omr_code = Column(String(3), nullable=False)
    omr_quest = Column(Integer, nullable=False)
    omr_answer = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<OMRTest omrid={self.omrid} uuid={self.uuid} code={self.omr_code}>"


class OCRTest(Base):
    """Luu dap an chu viet tay da duoc tai len."""

    __tablename__ = "ocr_test"

    ocrid = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(Integer, ForeignKey("users.uuid"), nullable=False)
    ocr_name = Column(String, nullable=False)
    ocr_answer = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<OCRTest ocrid={self.ocrid} uuid={self.uuid} name={self.ocr_name}>"
