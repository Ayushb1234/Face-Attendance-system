# backend/app/db/models.py
from datetime import datetime, date
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, Date, DateTime, Float
from ..deps import Base  # Base is defined in deps.py

class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)         # user_id (roll or slug)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    roll_no: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    dept: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

class Attendance(Base):
    __tablename__ = "attendance"
    # composite primary key: (user_id, day)
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)

    # status fields
    present: Mapped[bool] = mapped_column(Boolean, default=False)     # True=Present, False=Absent
    status: Mapped[str] = mapped_column(String(1), default="A")       # "P" or "A"

    # metadata
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    device_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    first_seen: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
