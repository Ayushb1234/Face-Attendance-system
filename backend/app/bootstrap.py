from pathlib import Path
import numpy as np, cv2
from .services.embedder import FaceEmbedder
from .services.matcher import Matcher
from .db import crud
from .deps import SessionLocal

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _nice_name(stem: str) -> str:
    return stem.replace("_"," ").replace("-"," ").strip().title()

def _user_id(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+","-", name.lower()).strip("-")

def enroll_dataset(subdir: str = "dataset") -> int:
    root = (Path(__file__).resolve().parents[1] / subdir)
    if not root.exists():
        return 0
    emb = FaceEmbedder()
    match = Matcher(dim=512)
    added = 0
    with SessionLocal() as db:
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS: 
                continue
            data = np.fromfile(str(p), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None: 
                continue
            faces = emb.run(bgr)
            if not faces: 
                continue
            face = faces[0]
            name = _nice_name(p.stem)
            uid = _user_id(name)
            match.add_embedding(uid, face["embedding"], {"source":"dataset","file":p.name})
            crud.ensure_user(db, user_id=uid, name=name)
            added += 1
    return added
