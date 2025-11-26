from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pandas as pd, datetime as dt, os
from fastapi import FastAPI
from sqlalchemy.orm import Session
from ..deps import get_db  # not directly usable outside req cycle
from ..deps import SessionLocal  # use session factory
from ..db import crud

def schedule_jobs(app: FastAPI):
    sch = AsyncIOScheduler(timezone="Asia/Kolkata")

    @sch.scheduled_job("cron", hour=23, minute=59)
    def daily_export():
        day = dt.date.today()
        with SessionLocal() as db:
            rows = crud.get_attendance_for_day(db, day)
        if not rows:
            return
        df = pd.DataFrame(rows)
        out_dir = "/data/exports"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"attendance_{day.isoformat()}.xlsx")
        df.to_excel(out_path, index=False)

    sch.start()
