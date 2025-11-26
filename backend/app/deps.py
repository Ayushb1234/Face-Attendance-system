from __future__ import annotations

from typing import Optional, Set
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from .settings import Settings

# --------------------------
# Settings / DB engine
# --------------------------
settings = Settings()

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# --------------------------
# Optional Qdrant wiring
# matcher.py imports: is_qdrant_enabled, qdrant, COLLECTION
# --------------------------
COLLECTION: str = getattr(settings, "QDRANT_COLLECTION", "faces")

qdrant: Optional["QdrantClient"] = None
_qmodels = None
try:
    if getattr(settings, "QDRANT_URL", None):
        from qdrant_client import QdrantClient, models as qmodels  # type: ignore
        qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=getattr(settings, "QDRANT_API_KEY", None))
        _qmodels = qmodels
except Exception:
    # keep qdrant=None if client not installed or config invalid
    qdrant = None
    _qmodels = None

def is_qdrant_enabled() -> bool:
    return qdrant is not None

def init_vector_collection(dim: int = 512) -> None:
    """Create the collection in Qdrant if configured. No-op for local matcher."""
    if not qdrant or not _qmodels:
        return
    try:
        qdrant.get_collection(COLLECTION)
        return
    except Exception:
        pass
    try:
        # recreate if it exists with wrong schema; otherwise create
        qdrant.recreate_collection(
            COLLECTION,
            vectors_config=_qmodels.VectorParams(size=dim, distance=_qmodels.Distance.COSINE),
        )
    except Exception:
        try:
            qdrant.create_collection(
                COLLECTION,
                vectors_config=_qmodels.VectorParams(size=dim, distance=_qmodels.Distance.COSINE),
            )
        except Exception:
            # don't hard-fail app startup for vector backend issues
            pass

# --------------------------
# DB bootstrap + legacy SQLite hot-patch
# --------------------------
def init_db() -> None:
    """
    Create tables and hot-patch legacy SQLite DBs to add any missing columns.
    Prevents errors like 'no such column: attendance.day/present'.
    """
    # register models
    from .db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # SQLite column add (safe no-ops if already there)
    if settings.DATABASE_URL.startswith("sqlite"):
        with engine.begin() as conn:
            def cols(table: str) -> Set[str]:
                return {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}

            if conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='attendance'"
            ).fetchone():
                existing = cols("attendance")

                def ensure(name: str, ddl: str) -> None:
                    if name not in existing:
                        conn.exec_driver_sql(f"ALTER TABLE attendance ADD COLUMN {name} {ddl}")

                ensure("day", "DATE")
                ensure("present", "BOOLEAN DEFAULT 0")
                ensure("status", "TEXT DEFAULT 'A'")
                ensure("confidence", "REAL")
                ensure("device_id", "TEXT")
                ensure("first_seen", "DATETIME")
                ensure("last_seen", "DATETIME")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
