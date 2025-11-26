# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .deps import settings, init_db

# Optional helpers (Qdrant)
try:
    from .deps import init_vector_collection, is_qdrant_enabled
except Exception:  # safe no-ops if not provided
    def init_vector_collection(): return None
    def is_qdrant_enabled() -> bool: return False

# Routers
from .routers import enroll, match, attendance

# Optional jobs/bootstrap (don’t crash if missing)
try:
    from .jobs.export_excel import schedule_jobs
except Exception:
    def schedule_jobs(app):  # type: ignore
        return

try:
    from .bootstrap import enroll_dataset
except Exception:
    def enroll_dataset(*_args, **_kwargs):  # type: ignore
        return 0

app = FastAPI(title="Face Attendance API", version="0.1.0")

def _cors_list():
    # Preferred: list from Settings.CORS_ORIGINS
    cors = getattr(settings, "CORS_ORIGINS", None)
    if isinstance(cors, (list, tuple)) and cors:
        return list(cors)
    # Legacy fallback: comma string from env ALLOW_ORIGINS
    s = os.getenv("ALLOW_ORIGINS", "")
    allow = [o.strip() for o in s.split(",") if o.strip()]
    return allow or ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire routes
app.include_router(enroll.router)
app.include_router(match.router)
app.include_router(attendance.router)

@app.get("/", tags=["meta"])
def root():
    return {"ok": True, "service": "face-attendance", "version": "0.1.0"}

@app.on_event("startup")
def _startup():
    # DB + (optional) vector backend + (optional) bootstrap + (optional) jobs
    init_db()
    if is_qdrant_enabled():
        try: init_vector_collection()
        except Exception: pass
    try:
        n = enroll_dataset("dataset")
        if n: print(f"Bootstrapped {n} embeddings from /backend/dataset")
    except Exception as e:
        print("Dataset bootstrap skipped:", e)
    try:
        schedule_jobs(app)
    except Exception:
        pass
