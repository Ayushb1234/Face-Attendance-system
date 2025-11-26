import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceEmbedder:
    """
    Uses InsightFace FaceAnalysis (buffalo_l) which includes RetinaFace detector + ArcFace embeddings.
    normed_embedding is already L2-normalized (good for cosine).
    """
    def __init__(self, det_size=(640, 640), providers=("CPUExecutionProvider",)):
        self.app = FaceAnalysis(name="buffalo_l", providers=list(providers))
        self.app.prepare(ctx_id=0, det_size=det_size)

    def run(self, bgr: np.ndarray):
        """
        Returns list of dicts: [{bbox, kps, embedding (float32[512])}, ...]
        """
        faces = self.app.get(bgr)
        out = []
        for f in faces:
            out.append({
                "bbox": [int(x) for x in f.bbox],
                "kps": np.array(f.kps).astype(float).tolist(),
                "embedding": f.normed_embedding.astype(np.float32),
            })
        return out
