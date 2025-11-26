# backend/app/db/crud.py
from __future__ import annotations
from datetime import date, datetime
from sqlalchemy import select
from sqlalchemy.orm import Session
from .models import User, Attendance

# --- users ---

def ensure_user(
    db: Session,
    user_id: str,
    name: str | None = None,
    roll_no: str | None = None,
    dept: str | None = None,
):
    """Create the user if missing; lightly update fields if provided."""
    u = db.get(User, user_id)
    if u:
        changed = False
        if name and (not u.name or u.name.startswith("User ")):
            u.name = name; changed = True
        if roll_no and not u.roll_no:
            u.roll_no = roll_no; changed = True
        if dept and not u.dept:
            u.dept = dept; changed = True
        if changed:
            db.commit(); db.refresh(u)
        return u

    u = User(id=user_id, name=name or f"User {user_id}", roll_no=roll_no, dept=dept, active=True)
    db.add(u); db.commit(); db.refresh(u)
    return u

# --- attendance ---

def mark_present(
    db: Session,
    user_id: str,
    day: date,
    confidence: float | None = None,
    device_id: str | None = None,
):
    """
    Upsert attendance row for (user_id, day):
      - set present=True / status='P'
      - keep the highest confidence
      - update last_seen (and first_seen if empty)
    """
    # make sure user exists
    ensure_user(db, user_id=user_id)

    stmt = select(Attendance).where(
        Attendance.user_id == user_id,
        Attendance.day == day,
    )
    rec = db.execute(stmt).scalar_one_or_none()
    now = datetime.now()

    if rec is None:
        rec = Attendance(
            user_id=user_id,
            day=day,
            present=True,
            status="P",
            confidence=confidence,
            device_id=device_id,
            first_seen=now,
            last_seen=now,
        )
        db.add(rec)
        db.commit(); db.refresh(rec)
        return rec

    # update existing
    rec.present = True
    rec.status = "P"
    if confidence is not None:
        prev = rec.confidence or 0.0
        if confidence > prev:
            rec.confidence = confidence
    rec.device_id = rec.device_id or device_id
    rec.first_seen = rec.first_seen or now
    rec.last_seen = now
    db.commit(); db.refresh(rec)
    return rec

def mark_absent_for_all_if_missing(db: Session, day: date):
    """
    Optional helper: for each active user without a row for 'day', insert Absent.
    """
    users = db.execute(select(User).where(User.active == True)).scalars().all()
    for u in users:
        stmt = select(Attendance).where(Attendance.user_id == u.id, Attendance.day == day)
        if db.execute(stmt).scalar_one_or_none() is None:
            db.add(Attendance(user_id=u.id, day=day, present=False, status="A"))
    db.commit()
