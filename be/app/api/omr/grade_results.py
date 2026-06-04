from fastapi import APIRouter

from .shared import *  # noqa: F403 - route modules share the existing helper surface during the split

router = APIRouter()


@router.get("/assignments/{uid}/{aid}/grade-results")
async def list_assignment_grade_results(uid: int, aid: int, db: Session = Depends(get_db)):
    uid = _validate_uid(uid)
    assignment = (
        db.query(OMRAssignment)
        .filter(OMRAssignment.uuid == uid, OMRAssignment.aid == int(aid))
        .first()
    )
    if not assignment:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài thi")

    records = (
        db.query(OMRGradeResult)
        .filter(OMRGradeResult.uuid == uid, OMRGradeResult.aid == int(aid))
        .order_by(OMRGradeResult.created_at.desc(), OMRGradeResult.grid.desc())
        .all()
    )
    return JSONResponse(content={"grade_results": [_serialize_grade_result(r) for r in records]})


@router.delete("/assignments/{uid}/{aid}/grade-results/{grid}")
async def delete_assignment_grade_result(uid: int, aid: int, grid: int, db: Session = Depends(get_db)):
    uid = _validate_uid(uid)
    record = (
        db.query(OMRGradeResult)
        .filter(
            OMRGradeResult.uuid == uid,
            OMRGradeResult.aid == int(aid),
            OMRGradeResult.grid == int(grid),
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả chấm")

    db.delete(record)
    db.commit()
    return JSONResponse(content={"message": "Đã xóa kết quả chấm"})
