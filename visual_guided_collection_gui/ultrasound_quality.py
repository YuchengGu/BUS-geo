from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.segmentation import random_walker


@dataclass(frozen=True)
class QualityFeatures:
    D: float
    E: float
    C: float
    S: float


@dataclass(frozen=True)
class QualityNormalization:
    d_min: float = 0.0
    d_max: float = 1.0
    e_min: float = 0.0
    e_max: float = 8.0
    c_min: float = 0.0
    c_max: float = 128.0
    c_target: float = 64.0
    s_min: float = 0.0
    s_max: float = 2.0


@dataclass(frozen=True)
class QualityScore:
    features: QualityFeatures
    normalized: np.ndarray
    weighted_vector: np.ndarray
    Q: float


def _to_gray_uint8(frame: np.ndarray, *, max_size: int | None = None) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] >= 3:
            image = (
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]
            )
        else:
            raise ValueError(f"unsupported ultrasound frame shape: {image.shape}")
    if image.ndim != 2:
        raise ValueError(f"ultrasound frame must be 2D or RGB-like, got {image.shape}")

    image = image.astype(float, copy=False)
    if image.size == 0:
        raise ValueError("ultrasound frame is empty")
    if np.nanmax(image) <= 1.0:
        image = image * 255.0
    image = np.clip(np.nan_to_num(image), 0, 255).astype(np.uint8)
    return _resize_max_size(image, max_size=max_size)


def _resize_max_size(image: np.ndarray, *, max_size: int | None) -> np.ndarray:
    if max_size is None or max_size <= 0:
        return image
    height, width = image.shape
    longest = max(height, width)
    if longest <= max_size:
        return image
    scale = float(max_size) / float(longest)
    resized = ndimage.zoom(image.astype(float, copy=False), zoom=(scale, scale), order=1)
    return np.clip(resized, 0, 255).astype(np.uint8)


def _shannon_entropy(frame: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(frame, bins=bins, range=(0, 256), density=True)
    hist = hist[hist > 0.0]
    return float(-np.sum(hist * np.log2(hist)))


def _speckle_index(frame: np.ndarray, win: int = 9, dark_abs: float = 10.0) -> float:
    image = frame.astype(float, copy=False)
    mu = ndimage.uniform_filter(image, size=win, mode="reflect")
    mean_sq = ndimage.uniform_filter(image * image, size=win, mode="reflect")
    std = np.sqrt(np.maximum(mean_sq - mu * mu, 0.0))
    mask = mu > dark_abs
    if not np.any(mask):
        return 2.0
    return float(np.mean(std[mask] / (mu[mask] + 1e-8)))


def _random_walker_confidence_map(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(float, copy=False)
    norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    markers = np.zeros_like(norm, dtype=np.uint8)
    top_end = max(2, min(10, norm.shape[0] // 3))
    bottom_start = max(top_end + 1, norm.shape[0] - max(2, norm.shape[0] // 8))
    markers[:top_end, :] = 1
    markers[bottom_start:, :] = 2
    try:
        labels = random_walker(norm, markers, beta=25, tol=5e-2, mode="cg_j")
    except Exception:
        labels = random_walker(norm, markers, beta=25, mode="bf")
    return (labels == 1).astype(np.float32)


def _fast_confidence_map(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(float, copy=False)
    norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    attenuation = ndimage.gaussian_filter1d(norm, sigma=2.0, axis=0)
    cumulative = np.cumsum(attenuation, axis=0)
    cumulative = cumulative / (np.max(cumulative) + 1e-8)
    confidence = np.exp(-3.0 * cumulative)
    top_boost = np.linspace(1.0, 0.35, frame.shape[0], dtype=float)[:, None]
    return np.clip(confidence * top_boost, 0.0, 1.0).astype(np.float32)


def _confidence_response(frame: np.ndarray, *, method: str) -> float:
    image = frame.astype(float, copy=False)
    if method == "fast":
        conf = _fast_confidence_map(frame)
    elif method == "random_walker":
        conf = _random_walker_confidence_map(frame)
    else:
        raise ValueError("confidence_method must be 'fast' or 'random_walker'")
    dark_abs = 10.0
    bright_high = float(np.percentile(image, 90))
    if bright_high <= dark_abs:
        bright = np.zeros_like(image, dtype=float)
    else:
        bright = np.clip((image - dark_abs) / (bright_high - dark_abs + 1e-8), 0.0, 1.0) ** 0.5
    return float(np.sum(conf * bright) / (np.sum(conf) + 1e-8))


def extract_quality_features(
    frame: np.ndarray,
    *,
    max_size: int | None = 160,
    speckle_max_size: int | None = 160,
    confidence_max_size: int | None = None,
    confidence_method: str = "fast",
) -> QualityFeatures:
    gray = _to_gray_uint8(frame, max_size=max_size)
    confidence_gray = gray if confidence_max_size == max_size else _to_gray_uint8(frame, max_size=confidence_max_size)
    speckle_gray = gray if speckle_max_size == max_size else _to_gray_uint8(frame, max_size=speckle_max_size)
    return QualityFeatures(
        D=_confidence_response(confidence_gray, method=confidence_method),
        E=_shannon_entropy(gray),
        C=float(np.std(gray.astype(float, copy=False))),
        S=_speckle_index(speckle_gray),
    )


class UltrasoundQualityScorer:
    def __init__(
        self,
        *,
        normalization: QualityNormalization | None = None,
        weights: np.ndarray | None = None,
        max_size: int | None = 160,
        speckle_max_size: int | None = 160,
        confidence_max_size: int | None = None,
        confidence_method: str = "fast",
    ) -> None:
        self.normalization = normalization or QualityNormalization()
        self.max_size = max_size
        self.speckle_max_size = speckle_max_size
        self.confidence_max_size = max_size if confidence_max_size is None else confidence_max_size
        if confidence_method not in {"fast", "random_walker"}:
            raise ValueError("confidence_method must be 'fast' or 'random_walker'")
        self.confidence_method = confidence_method
        self.weights = np.asarray(
            weights if weights is not None else [0.4833, 0.1620, 0.0794, 0.2754],
            dtype=float,
        )
        if self.weights.shape != (4,):
            raise ValueError(f"quality weights must have shape (4,), got {self.weights.shape}")
        self.weights = self.weights / (np.sum(self.weights) + 1e-12)

    def normalize(self, features: QualityFeatures) -> np.ndarray:
        n = self.normalization
        d = _safe_ratio(features.D - n.d_min, n.d_max - n.d_min)
        e = _safe_ratio(features.E - n.e_min, n.e_max - n.e_min)
        c = 1.0 - abs(features.C - n.c_target) / (n.c_max - n.c_min + 1e-12)
        s = _safe_ratio(n.s_max - features.S, n.s_max - n.s_min)
        return np.clip(np.array([d, e, c, s], dtype=float), 0.0, 1.0)

    def score_features(self, features: QualityFeatures) -> QualityScore:
        normalized = self.normalize(features)
        weighted = self.weights * normalized
        positive = self.weights
        negative = np.zeros(4, dtype=float)
        d_pos = float(np.linalg.norm(weighted - positive))
        d_neg = float(np.linalg.norm(weighted - negative))
        q = d_neg / (d_pos + d_neg + 1e-12)
        return QualityScore(features=features, normalized=normalized, weighted_vector=weighted, Q=float(q))

    def score_frame(self, frame: np.ndarray) -> QualityScore:
        return self.score_features(
            extract_quality_features(
                frame,
                max_size=self.max_size,
                speckle_max_size=self.speckle_max_size,
                confidence_max_size=self.confidence_max_size,
                confidence_method=self.confidence_method,
            )
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)
