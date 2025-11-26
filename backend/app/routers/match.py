# backend/app/routers/match.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import base64, numpy as np, cv2, datetime as dt, time, traceback
from sqlalchemy.orm import Session

from ..services.embedder import FaceEmbedder
from ..services.matcher import Matcher
from ..db import crud
from ..deps import get_db

# optional imports (crop + liveness). Safe if not present.
try:
    from ..services.detector import Detector as _Det  # for crop helper
except Exception:
    _Det = None  # type: ignore

try:
    from ..services.liveness import LivenessGuard
except Exception:
    LivenessGuard = None  # type: ignore

router = APIRouter(prefix="/match", tags=["match"])

# ---- services (singletons) ----
_embedder = FaceEmbedder()
_matcher = Matcher()
_guard = None
if LivenessGuard is not None:
    _guard = LivenessGuard(
        onnx_model_path=None,
        max_abs_yaw=70.0,
        max_abs_pitch=70.0,
        min_motion_std=0.1,
    )

# ---- thresholds / behavior (tune per environment) ----
THRESHOLD = 0.30          # cosine similarity threshold (ArcFace cos)
COOLDOWN_SEC = 120        # won't re-mark same user within this window
REQUIRE_LIVENESS = False  # gate matching with liveness when True
MIN_INTERVAL = 0.20       # throttle: don't process faster than N seconds/frame (~5 FPS)

def _area(bbox) -> float:
    try:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
    except Exception:
        return 0.0

def _pick_largest_face(faces: list[dict]) -> dict:
    return max(faces, key=lambda f: _area(f.get("bbox", (0, 0, 0, 0))))

@router.websocket("/stream")
async def match_stream(ws: WebSocket, db: Session = Depends(get_db)):
    await ws.accept()
    last_seen: dict[str, dt.datetime] = {}
    last_proc_ts = 0.0

    try:
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict):
                await ws.send_json({"matched": False, "error": "bad_message"})
                continue

            # device id (stored in attendance for provenance)
            device_id = (msg.get("device_id") or "web").strip()

            # throttle to avoid CPU spikes
            now_ts = time.time()
            if now_ts - last_proc_ts < MIN_INTERVAL:
                continue
            last_proc_ts = now_ts

            try:
                # --- decode frame ---
                frame_str = msg.get("frame")
                if not frame_str or not isinstance(frame_str, str):
                    await ws.send_json({"matched": False, "error": "no_frame"})
                    continue

                b64 = frame_str.split(",")[-1]
                data = np.frombuffer(base64.b64decode(b64), np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    await ws.send_json({"matched": False, "error": "decode_failed"})
                    continue

                # --- detect/embed ---
                faces = _embedder.run(bgr)  # returns list of dicts: bbox, kps, embedding
                if not faces:
                    await ws.send_json({"matched": False})
                    continue

                face = _pick_largest_face(faces)
                bbox = tuple(face.get("bbox", (0, 0, 0, 0)))
                kps = face.get("kps")

                # --- liveness (optional) ---
                live_info = None
                if REQUIRE_LIVENESS and _guard is not None:
                    crop = None
                    if _Det is not None and hasattr(_Det, "crop_face"):
                        try:
                            crop = _Det.crop_face(bgr, bbox, margin=0.2, square=True)  # type: ignore[attr-defined]
                        except Exception:
                            crop = None
                    try:
                        live_info = _guard.check(bgr, bbox, kps, crop=crop)  # type: ignore[union-attr]
                    except Exception:
                        live_info = {"pass": False, "reasons": ["liveness_error"]}
                    if not live_info.get("pass", False):
                        await ws.send_json({"matched": False, "liveness": live_info, "reason": "liveness_failed"})
                        continue

                # --- search/vector match ---
                emb = face.get("embedding")
                if emb is None:
                    await ws.send_json({"matched": False, "error": "no_embedding"})
                    continue

                res = _matcher.top1(emb)
                score = float(res["score"]) if res and "score" in res else 0.0
                if not res or score < THRESHOLD:
                    await ws.send_json({"matched": False, "score": score, "liveness": live_info})
                    continue

                user_id = str(res.get("user_id"))

                # --- attendance upsert with cooldown ---
                now = dt.datetime.now()
                prev = last_seen.get(user_id)
                if prev is None or (now - prev).total_seconds() > COOLDOWN_SEC:
                    try:
                        crud.mark_present(
                            db,
                            user_id=user_id,
                            day=now.date(),
                            confidence=score,
                            device_id=device_id,
                        )
                    except Exception:
                        # don't kill the socket if DB has an issue
                        traceback.print_exc()
                    last_seen[user_id] = now

                await ws.send_json(
                    {"matched": True, "user_id": user_id, "score": score, "liveness": live_info}
                )

            except Exception as e:
                # keep WS alive; surface error to client for debugging
                traceback.print_exc()
                try:
                    await ws.send_json({"matched": False, "error": "server_error", "detail": str(e)[:200]})
                except Exception:
                    # ignore send failures and continue loop
                    pass

    except WebSocketDisconnect:
        # client closed the socket; exit gracefully
        return
