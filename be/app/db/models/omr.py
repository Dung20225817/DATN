# app/db/ocr_tables.py

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from datetime import datetime

from app.db.session import Base


class OMRTest(Base):
    """Luu thong tin de OMR do giao vien tao."""

    __tablename__ = "omr_test"

    omrid = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(Integer, ForeignKey("users.uuid"), nullable=False)
    omr_name = Column(String, nullable=False)
    omr_code = Column(String(3), nullable=False)
    omr_quest = Column(Integer, nullable=False)
    omr_answer = Column(JSON, nullable=False)
    template_image = Column(String, nullable=True)
    info_fields = Column(JSON, nullable=True)
    options = Column(Integer, nullable=True)
    rows_per_block = Column(Integer, nullable=True)
    student_id_digits = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<OMRTest omrid={self.omrid} uuid={self.uuid} code={self.omr_code}>"


class OMRAssignment(Base):
    """Luu bai thi OMR tren giao dien mobile, gom bo dap an theo ma de va lich su cham."""

    __tablename__ = "omr_assignment"

    aid = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(Integer, ForeignKey("users.uuid"), nullable=False)
    title = Column(String, nullable=False)
    created_at_raw = Column(String, nullable=True)
    created_at_label = Column(String, nullable=True)
    question_count = Column(Integer, nullable=False, default=40)
    total_points = Column(Integer, nullable=False, default=10)
    graded_count = Column(Integer, nullable=False, default=0)
    answer_sets = Column(JSON, nullable=False, default=list)
    active_code = Column(String, nullable=True)
    last_result = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<OMRAssignment aid={self.aid} uuid={self.uuid} title={self.title}>"


class OMRGradeResult(Base):
    """Luu metadata ket qua cham OMR, file nhi phan van nam trong storage/uploads."""

    __tablename__ = "omr_grade_result"

    grid = Column(Integer, primary_key=True, autoincrement=True)
    aid = Column(Integer, ForeignKey("omr_assignment.aid"), nullable=True)
    uuid = Column(Integer, ForeignKey("users.uuid"), nullable=True)
    omrid = Column(Integer, ForeignKey("omr_test.omrid"), nullable=True)
    source = Column(String, nullable=True)
    file_name = Column(String, nullable=True)
    student_id = Column(String, nullable=True)
    exam_code = Column(String, nullable=True)
    score = Column(String, nullable=True)
    result_image = Column(String, nullable=True)
    sid_crop_image = Column(String, nullable=True)
    mcq_crop_image = Column(String, nullable=True)
    bubble_confidence_json = Column(String, nullable=True)
    result_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<OMRGradeResult grid={self.grid} aid={self.aid} student_id={self.student_id}>"
