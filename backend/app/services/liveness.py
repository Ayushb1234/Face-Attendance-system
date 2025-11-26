"""
Lightweight liveness gates:
  1) Geometry/quality: face size, in-frame, not extreme pose.
  2) Pose gate: yaw/pitch from 5-point landmarks via solvePnP.
  3) Motion gate: small natural motion across a sliding window (anti static photo).
  4) Optional texture model (ONNX binary classifier live vs spoof).

Use:
  guard = LivenessGuard()
  verdict = guard.check(bgr, bbox, kps, crop=face_crop)
  if verdict["pass"]: accept match; else reject.

Dependencies: numpy, opencv, onnxruntime (optional).
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Optional
import numpy as np
import cv2

try:
    import onnxruntime as ort  # optional
except Exception:  # pragma: no cover
    ort = None  # type: ignore


@dataclass
class Pose:
    yaw: float
    pitch: float
    roll: float


@dataclass
class LivenessVerdict:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    pose: Optional[Pose] = None
    texture_score: Optional[float] = None
    motion_score: Optional[float] = None


class PoseEstimator5pt:
    """
    Estimate yaw/pitch/roll using 5-point landmarks:
      kps order expected: [left_eye, right_eye, nose, left_mouth, right_mouth]
    Uses a simple canonical 3D face model; absolute values aren't perfect,
    but thresholds like |yaw|<25°, |pitch|<25° work reliably for gating.
    """

    def __init__(self):
        # Canonical 3D points (unit-mm-ish; relative scale is fine)
        self.model_points = np.array(
            [
                (-30.0, 0.0, -30.0),   # left eye center
                (30.0, 0.0, -30.0),    # right eye center
                (0.0, 0.0, 0.0),       # nose tip
                (-20.0, -25.0, -30.0), # left mouth corner
                (20.0, -25.0, -30.0),  # right mouth corner
            ],
            dtype=np.float64,
        )

    def estimate(self, img_shape: Tuple[int, int], kps: List[List[float]]) -> Pose:
        h, w = img_shape[:2]
        focal = w  # rough fx=fy
        cam_matrix = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1))  # assume no distortion

        pts2d = np.array(kps, dtype=np.float64)
        assert pts2d.shape == (5, 2), "Expected 5-point landmarks"

        ok, rvec, tvec = cv2.solvePnP(self.model_points, pts2d, cam_matrix, dist, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            # fallback small angles
            return Pose(yaw=0.0, pitch=0.0, roll=0.0)

        R, _ = cv2.Rodrigues(rvec)
        # Convert rotation matrix to Euler angles (ZYX -> yaw,pitch,roll)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        else:
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
            roll = 0.0
        return Pose(yaw=float(yaw), pitch=float(pitch), roll=float(roll))


class TextureClassifierONNX:
    """
    Optional binary liveness model (ONNX). Expects a 112x112 RGB or BGR crop,
    normalized to 0..1. Output: live_prob in [0,1].
    If no model_path or onnxruntime missing -> always returns None (skip).
    """

    def __init__(self, model_path: Optional[str] = None, input_name: Optional[str] = None, output_name: Optional[str] = None):
        self.session = None
        self.input_name = input_name
        self.output_name = output_name
        if model_path and ort is not None:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            if self.input_name is None:
                self.input_name = self.session.get_inputs()[0].name
            if self.output_name is None:
                self.output_name = self.session.get_outputs()[0].name

    def predict_live_prob(self, crop_bgr: np.ndarray) -> Optional[float]:
        if self.session is None:
            return None
        img = cv2.resize(crop_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        img = img[:, :, ::-1]  # BGR->RGB
        img = img.astype(np.float32) / 255.0
        x = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        live_prob = float(out.ravel()[0])
        return live_prob


class MotionGate:
    """
    Keeps a short history of face centers to ensure tiny natural motion over time.
    Rejects perfectly static prints/screens. Windowed stddev is used as proxy.
    """

    def __init__(self, window: int = 12):
        self.window = window
        self.centers: Deque[Tuple[float, float]] = deque(maxlen=window)

    def update(self, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.centers.append((cx, cy))
        if len(self.centers) < 4:
            return 0.0
        arr = np.array(self.centers, dtype=np.float32)
        # std magnitude (pixels)
        std = float(np.linalg.norm(np.std(arr, axis=0)))
        return std


class LivenessGuard:
    """
    Combine geometry/pose/motion/texture checks into a single verdict.
    Thresholds are conservative defaults; tweak per camera & environment.
    """

    def __init__(
        self,
        min_face_side: int = 80,          # px on the long side
        max_abs_yaw: float = 25.0,        # degrees
        max_abs_pitch: float = 25.0,      # degrees
        min_motion_std: float = 0.8,      # px std over window
        texture_live_threshold: float = 0.6,  # live probability
        onnx_model_path: Optional[str] = None,
    ):
        self.pose_est = PoseEstimator5pt()
        self.motion = MotionGate(window=12)
        self.texture = TextureClassifierONNX(onnx_model_path) if onnx_model_path else TextureClassifierONNX(None)
        self.min_face_side = min_face_side
        self.max_abs_yaw = max_abs_yaw
        self.max_abs_pitch = max_abs_pitch
        self.min_motion_std = min_motion_std
        self.texture_live_threshold = texture_live_threshold

    def check(
        self,
        bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        kps: List[List[float]],
        crop: Optional[np.ndarray] = None,
    ) -> Dict:
        verdict = LivenessVerdict(passed=True, reasons=[])

        # 1) Geometry / size gate
        x1, y1, x2, y2 = bbox
        side = max(x2 - x1, y2 - y1)
        if side < self.min_face_side:
            verdict.passed = False
            verdict.reasons.append(f"face_too_small:{side}px<{self.min_face_side}px")

        # 2) Pose gate
        pose = self.pose_est.estimate(bgr.shape[:2], kps)
        verdict.pose = pose
        if abs(pose.yaw) > self.max_abs_yaw:
            verdict.passed = False
            verdict.reasons.append(f"yaw_exceeds:{pose.yaw:.1f}deg")
        if abs(pose.pitch) > self.max_abs_pitch:
            verdict.passed = False
            verdict.reasons.append(f"pitch_exceeds:{pose.pitch:.1f}deg")

        # 3) Motion gate
        motion_std = self.motion.update(bbox)
        verdict.motion_score = motion_std
        if motion_std < self.min_motion_std:
            verdict.passed = False
            verdict.reasons.append(f"low_motion:{motion_std:.2f}px")

        # 4) Texture gate (optional)
        if self.texture.session is not None:
            if crop is None:
                crop = bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            live_prob = self.texture.predict_live_prob(crop)
            verdict.texture_score = live_prob
            if live_prob is not None and live_prob < self.texture_live_threshold:
                verdict.passed = False
                verdict.reasons.append(f"texture_live_low:{live_prob:.2f}")

        return {
            "pass": verdict.passed,
            "reasons": verdict.reasons,
            "yaw": verdict.pose.yaw if verdict.pose else 0.0,
            "pitch": verdict.pose.pitch if verdict.pose else 0.0,
            "roll": verdict.pose.roll if verdict.pose else 0.0,
            "motion_std": verdict.motion_score,
            "texture_live": verdict.texture_score,
        }
