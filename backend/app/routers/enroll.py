from fastapi import APIRouter, UploadFile, Form, Depends, HTTPException
import numpy as np, cv2, time, re
from pathlib import Path
from sqlalchemy.orm import Session

from ..services.embedder import FaceEmbedder
from ..services.matcher import Matcher
from ..db import crud
from ..deps import get_db

router = APIRouter(prefix="/enroll", tags=["enroll"])
_embedder = FaceEmbedder()
_matcher = Matcher()

ROOT = Path(__file__).resolve().parents[2]   # .../backend
DATASET_DIR = ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-") or "user"

@router.post("")
async def enroll(
    user_id: str = Form(...),
    image: UploadFile | None = None,
    name: str | None = Form(None),
    roll_no: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if image is None:
        raise HTTPException(400, "image file is required")

    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "invalid image")

    faces = _embedder.run(bgr)
    if len(faces) != 1:
        raise HTTPException(400, f"expected exactly 1 face, found {len(faces)}")

    emb = faces[0]["embedding"]
    crud.ensure_user(db, user_id=user_id, name=name, roll_no=roll_no)

    # 1) add to local vector index (or qdrant if enabled)
    _matcher.add_embedding(user_id, emb, {"source": "enroll_api"})

    # 2) persist image under backend/dataset
    safe = _slug(name or user_id)
    out = DATASET_DIR / f"{safe}_{user_id}_{int(time.time())}.jpg"
    cv2.imwrite(str(out), bgr)

    return {"ok": True, "user_id": user_id, "name": name, "roll_no": roll_no, "saved_to": str(out)}
