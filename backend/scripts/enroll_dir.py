# backend/scripts/enroll_dir.py
# Usage:
#   python backend/scripts/enroll_dir.py --dir "D:/faces_dataset"
# Supported filename patterns:
#   - "Ayush.jpg"                  -> name="Ayush", user_id="ayush"
#   - "CS-01_Ayush.png"            -> roll_no="CS-01", name="Ayush", user_id="CS-01"
#   - Folders also OK: dataset/CS-01_Ayush/1.jpg ...

from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np, cv2

# make "app" importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.embedder import FaceEmbedder
from app.services.matcher import Matcher
from app.deps import SessionLocal
from app.db import crud

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s.strip().lower()).strip("-")
    return s or "user"

def parse_name_roll(stem: str) -> tuple[str, Optional[str]]:
    """
    Return (name, roll_no?). Accepts 'CS-01_Ayush', 'CS01-Ayush', 'Ayush'.
    """
    s = stem.replace(".", " ").replace("__", "_").strip()
    s = s.replace("-", "_")
    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2 and re.match(r"^[A-Za-z]{1,5}\-?\d{1,4}$", parts[0]):
        roll = parts[0].upper()
        name = " ".join(parts[1:]).title()
        return name, roll
    # fallback: whole stem is the name
    name = stem.replace("_", " ").replace("-", " ").title()
    return name, None

def best_face(faces):
    # choose largest bbox area
    def area(b):
        return (b[2]-b[0])*(b[3]-b[1])
    return max(faces, key=lambda f: area(f["bbox"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Path to dataset root")
    ap.add_argument("--multi", action="store_true", help="Allow multiple embeddings per user (recommended)")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"❌ No such dir: {root}")
        sys.exit(1)

    embedder = FaceEmbedder()
    matcher = Matcher(dim=512)

    count_added = 0
    with SessionLocal() as db:
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue

            name, roll_no = parse_name_roll(p.stem)
            # user_id: prefer roll_no (stable), else slug of name
            user_id = roll_no or slugify(name)

            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"skip (bad image): {p}")
                continue

            faces = embedder.run(img)
            if not faces:
                print(f"skip (no face): {p}")
                continue

            face = best_face(faces)
            emb = face["embedding"]

            # write to vector index (local or qdrant, depending on env)
            matcher.add_embedding(user_id, emb, {"source": "bulk", "file": p.name})

            # ensure user in SQL with nice metadata
            crud.ensure_user(db, user_id=user_id, name=name, roll_no=roll_no)
            count_added += 1
            print(f"✔ enrolled {user_id} ({name}) from {p.name}")

    print(f"\n✅ Done. Embeddings added: {count_added}")
    print("Now start frontend, open camera — matches will mark Present and Excel will include the Name/ Roll No.")
    print("Excel: GET /attendance/export?date=today")

if __name__ == "__main__":
    main()
