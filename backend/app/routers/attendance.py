# backend/app/routers/attendance.py
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import date, datetime
import io
from openpyxl import Workbook

from ..deps import get_db
from ..db.models import User, Attendance
from ..db import crud

router = APIRouter(prefix="/attendance", tags=["attendance"])

@router.get("/today")
def today(db: Session = Depends(get_db)):
    d = date.today()
    users = db.execute(select(User).order_by(User.id)).scalars().all()
    recs = db.execute(select(Attendance).where(Attendance.day == d)).scalars().all()
    return {
        "date": d.isoformat(),
        "users": [{"id": u.id, "name": u.name, "roll": u.roll_no} for u in users],
        "attendance": [
            {
                "user_id": r.user_id,
                "status": r.status,
                "confidence": r.confidence,
                "device": r.device_id,
                "first_seen": r.first_seen,
                "last_seen": r.last_seen,
            }
            for r in recs
        ],
    }

@router.get("/export")
def export_attendance(day: str = Query("today"), db: Session = Depends(get_db)):
    d = date.today() if day == "today" else datetime.strptime(day, "%Y-%m-%d").date()

    # ensure everyone has a row for the day (Absent if not seen)
    crud.mark_absent_for_all_if_missing(db, d)

    users = db.execute(select(User).order_by(User.id)).scalars().all()
    att_rows = db.execute(select(Attendance).where(Attendance.day == d)).scalars().all()
    att_map = {r.user_id: r for r in att_rows}

    from openpyxl import Workbook
    import io

    wb = Workbook()
    ws = wb.active
    ws.title = d.isoformat()

    headers = ["Date", "User ID", "Roll No", "Name", "Status",
               "Confidence", "Device", "First Seen", "Last Seen"]
    ws.append(headers)  # header ALWAYS

    if not users and not att_rows:
        # write a friendly empty row so the sheet is not blank
        ws.append([d.isoformat(), "", "", "", "NO DATA", "", "", "", ""])
    else:
        # output users first (ensures A/P for all)
        for u in users:
            r = att_map.get(u.id)
            ws.append([
                d.isoformat(),
                u.id,
                u.roll_no or "",
                u.name or "",
                (r.status if r else "A"),
                ("" if not r or r.confidence is None else r.confidence),
                ("" if not r else (r.device_id or "")),
                ("" if not r or not r.first_seen else r.first_seen.strftime("%H:%M:%S")),
                ("" if not r or not r.last_seen else r.last_seen.strftime("%H:%M:%S")),
            ])
        # also include any attendance rows for users not in the users table (edge case)
        for uid, r in att_map.items():
            if any(u.id == uid for u in users):
                continue
            ws.append([
                d.isoformat(), uid, "", "", r.status,
                ("" if r.confidence is None else r.confidence),
                (r.device_id or ""),
                ("" if not r.first_seen else r.first_seen.strftime("%H:%M:%S")),
                ("" if not r.last_seen else r.last_seen.strftime("%H:%M:%S")),
            ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="attendance_{d.isoformat()}.xlsx"'},
    )
