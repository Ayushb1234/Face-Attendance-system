"""
Face detector + utility helpers.

We wrap InsightFace's FaceAnalysis ONLY for detection/landmarks so you can
use this in places where you don't need embeddings. Pair with embedder.py when
you want 512-d ArcFace vectors.

Returns:
  - bbox: [x1, y1, x2, y2] ints
  - kps:  5-point landmarks [[lx,ly],[rx,ry],[nose],[lmouth],[rmouth]]
  - score: detection confidence (float)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
from insightface.app import FaceAnalysis


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]
    kps: List[List[float]]
    score: float


class Detector:
    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        providers: Tuple[str, ...] = ("CPUExecutionProvider",),
    ):
        # allowed_modules keeps it lean (detection only)
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=list(providers),
            allowed_modules=["detection"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, bgr: np.ndarray, max_faces: int | None = None) -> List[DetectedFace]:
        faces = self.app.get(bgr)
        out: List[DetectedFace] = []
        for f in faces:
            bbox = tuple(int(x) for x in f.bbox)  # x1,y1,x2,y2
            kps = np.array(f.kps).astype(float).tolist()
            score = float(getattr(f, "det_score", 1.0))
            out.append(DetectedFace(bbox=bbox, kps=kps, score=score))
        # sort by area (desc) so face[0] is the biggest in frame
        out.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        if max_faces:
            out = out[:max_faces]
        return out

    @staticmethod
    def crop_face(
        bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.2,
        square: bool = True,
    ) -> np.ndarray:
        """
        Crop a padded (optionally square) face region for downstream tasks.
        """
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        if square:
            side = max(bw, bh)
            cx, cy = x1 + bw / 2, y1 + bh / 2
            x1 = int(cx - side / 2)
            y1 = int(cy - side / 2)
            x2 = int(cx + side / 2)
            y2 = int(cy + side / 2)

        # margin padding
        pad_x = int((x2 - x1) * margin)
        pad_y = int((y2 - y1) * margin)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = bgr[y1:y2, x1:x2]
        return crop

    @staticmethod
    def draw_debug(bgr: np.ndarray, det: DetectedFace) -> np.ndarray:
        """Quick visual overlay for debugging."""
        out = bgr.copy()
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x, y) in det.kps:
            cv2.circle(out, (int(x), int(y)), 2, (255, 0, 0), -1)
        return out
