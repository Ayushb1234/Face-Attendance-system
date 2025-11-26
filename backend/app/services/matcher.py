# backend/app/services/matcher.py
import os, json, uuid
from typing import Optional, Dict, Any, List
import numpy as np
from ..deps import is_qdrant_enabled, qdrant, COLLECTION

# ---------- Local persistent matcher (no external service) ----------
class _LocalIndex:
    """
    Stores embeddings and metadata to ./data/index.npz + ./data/meta.json
    - vectors: float32 [N, D]
    - ids:     list[str] (point ids)
    - user_ids:list[str]
    """
    def __init__(self, dim: int = 512, data_dir: str = "data"):
        self.dim = dim
        self.data_dir = data_dir
        self.npz_path = os.path.join(data_dir, "index.npz")
        self.meta_path = os.path.join(data_dir, "meta.json")
        os.makedirs(self.data_dir, exist_ok=True)

        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.ids: List[str] = []
        self.user_ids: List[str] = []
        self._load()

    def _load(self):
        if os.path.exists(self.npz_path):
            npz = np.load(self.npz_path)
            self.vectors = npz["vectors"].astype(np.float32)
        if os.path.exists(self.meta_path):
            meta = json.load(open(self.meta_path, "r", encoding="utf-8"))
            self.ids = meta.get("ids", [])
            self.user_ids = meta.get("user_ids", [])
        # sanity
        if len(self.ids) != self.vectors.shape[0] or len(self.user_ids) != self.vectors.shape[0]:
            # reset if mismatch
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            self.ids, self.user_ids = [], []

    def _save(self):
        np.savez_compressed(self.npz_path, vectors=self.vectors)
        json.dump({"ids": self.ids, "user_ids": self.user_ids}, open(self.meta_path, "w", encoding="utf-8"))

    def add(self, user_id: str, vector: np.ndarray, meta: Dict[str, Any] | None = None) -> str:
        assert vector.shape[-1] == self.dim
        pid = str(uuid.uuid4())
        v = vector.astype(np.float32)[None, :]
        self.vectors = np.vstack([self.vectors, v]) if self.vectors.size else v
        self.ids.append(pid)
        self.user_ids.append(user_id)
        self._save()
        return pid

    def top1(self, vector: np.ndarray) -> Optional[dict]:
        if self.vectors.shape[0] == 0:
            return None
        v = vector.astype(np.float32)
        # cosine sim; embeddings are L2-normalized by ArcFace, so dot == cosine
        sims = (self.vectors @ v).ravel()  # [N]
        idx = int(np.argmax(sims))
        return {"user_id": self.user_ids[idx], "score": float(sims[idx]), "id": self.ids[idx]}

# ---------- Qdrant matcher (kept for future) ----------
class _QdrantMatcher:
    def __init__(self):
        self.client = qdrant

    def add_embedding(self, user_id: str, vector: np.ndarray, meta: dict | None = None) -> str:
        from qdrant_client.http.models import PointStruct  # lazy import
        pid = str(uuid.uuid4())
        payload = {"user_id": user_id}
        if meta:
            payload.update(meta)
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=pid, vector=vector.astype(float).tolist(), payload=payload)]
        )
        return pid

    def top1(self, vector: np.ndarray) -> Optional[dict]:
        res = self.client.search(collection_name=COLLECTION, query_vector=vector.astype(float).tolist(), limit=1)
        if not res:
            return None
        p = res[0]
        return {"user_id": p.payload.get("user_id"), "score": float(p.score), "id": str(p.id)}

# ---------- Unified facade ----------
class Matcher:
    """
    Drop-in replacement previously used by the routers.
    Chooses backend based on MATCHER_BACKEND env (local by default).
    """
    def __init__(self, dim: int = 512):
        if is_qdrant_enabled():
            self.backend = _QdrantMatcher()
        else:
            self.backend = _LocalIndex(dim=dim)

    def add_embedding(self, user_id: str, vector: np.ndarray, meta: dict | None = None) -> str:
        return self.backend.add(user_id, vector, meta)  # type: ignore

    def top1(self, vector: np.ndarray) -> Optional[dict]:
        return self.backend.top1(vector)  # type: ignore
