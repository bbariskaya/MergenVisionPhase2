"""Domain types for Sprint 06 offline video tracking and identity reconciliation.

All timestamps are in nanoseconds internally; public API reports seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

EMBEDDING_DIM = 512


@dataclass(frozen=True)
class RecognitionObservation:
    detection_id: str
    frame: int
    pts_ns: int
    bbox: Tuple[float, float, float, float]
    detector_score: float
    embedding: Tuple[float, ...]
    embedding_quality: float = 0.0
    face_width: float = 0.0
    face_height: float = 0.0
    border_clip_fraction: float = 0.0
    alignment_valid: bool = True
    frontal_score: float = 0.0
    sharpness_score: float = 0.0
    top1_similarity: Optional[float] = None
    top2_similarity: Optional[float] = None
    similarity_margin: Optional[float] = None


@dataclass(frozen=True)
class TrackletEvidence:
    tracklet_id: str
    source_id: int
    start_frame: int
    end_frame: int
    start_pts_ns: int
    end_pts_ns: int
    observations: Tuple[RecognitionObservation, ...]

    def duration_ns(self) -> int:
        return max(0, self.end_pts_ns - self.start_pts_ns)


@dataclass(frozen=True)
class AppearanceInterval:
    start_pts_ns: int
    end_pts_ns: int

    def duration_ns(self) -> int:
        return max(0, self.end_pts_ns - self.start_pts_ns)


@dataclass(frozen=True)
class BestShotEvidence:
    frame: int
    pts_ns: int
    detection_id: str
    quality: float
    embedding_ref: Optional[int] = None


@dataclass(frozen=True)
class CanonicalVideoPerson:
    video_person_id: str
    face_id: Optional[str]
    status: str
    name: Optional[str]
    tracklet_ids: Tuple[str, ...]
    appearances: Tuple[AppearanceInterval, ...]
    first_seen: float
    last_seen: float
    total_duration: float
    final_confidence: float
    best_shot: BestShotEvidence


@dataclass(frozen=True)
class ReconciliationConfig:
    embedding_dim: int = EMBEDDING_DIM
    min_observations: int = 1
    min_face_size: float = 24.0
    max_border_clip_fraction: float = 0.25
    min_embedding_quality: float = 0.0
    min_frontal_score: float = 0.0
    min_temporal_separation_ns: int = 0
    top_k_observations: int = 5

    known_accept_top1_threshold: float = 0.40
    known_accept_margin_threshold: float = 0.10
    known_min_consistency: float = 0.80

    anonymous_match_top1_threshold: float = 0.35
    anonymous_match_margin_threshold: float = 0.08

    unknown_cluster_threshold: float = 0.30
    unknown_cluster_margin: float = 0.05

    appearance_gap_ns: int = 2_000_000_000


def _is_valid_observation(o: RecognitionObservation, cfg: ReconciliationConfig) -> bool:
    face_size = min(o.face_width, o.face_height)
    if face_size < cfg.min_face_size:
        return False
    if o.border_clip_fraction > cfg.max_border_clip_fraction:
        return False
    if o.embedding_quality < cfg.min_embedding_quality:
        return False
    if o.frontal_score < cfg.min_frontal_score:
        return False
    if len(o.embedding) != cfg.embedding_dim:
        return False
    if any(not v == v for v in o.embedding):
        return False
    return True


def _l2_normalize(vector: Sequence[float]) -> Tuple[float, ...]:
    norm_sq = sum(v * v for v in vector)
    if norm_sq <= 0.0:
        return tuple(vector)
    norm = norm_sq ** 0.5
    return tuple(v / norm for v in vector)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))


def _seconds_from_ns(ts_ns: int) -> float:
    return ts_ns / 1_000_000_000.0


def _select_diverse_observations(
    observations: Sequence[RecognitionObservation],
    cfg: ReconciliationConfig,
) -> Sequence[RecognitionObservation]:
    if not observations:
        return observations
    sorted_obs = sorted(observations, key=lambda o: o.pts_ns)
    selected: list[RecognitionObservation] = []
    last_pts: Optional[int] = None
    for o in sorted_obs:
        if last_pts is None or (o.pts_ns - last_pts) >= cfg.min_temporal_separation_ns:
            selected.append(o)
            last_pts = o.pts_ns
    return selected[: cfg.top_k_observations]


def build_tracklet_prototype(
    tracklet: TrackletEvidence,
    cfg: ReconciliationConfig,
) -> Optional[Tuple[float, ...]]:
    valid = [o for o in tracklet.observations if _is_valid_observation(o, cfg)]
    if len(valid) < cfg.min_observations:
        return None
    diverse = _select_diverse_observations(valid, cfg)
    if not diverse:
        return None
    embeddings = [_l2_normalize(o.embedding) for o in diverse]
    mean = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(cfg.embedding_dim)]
    return _l2_normalize(mean)


def gallery_top_scores(
    prototype: Sequence[float],
    gallery: dict[str, Sequence[float]],
) -> Tuple[Optional[str], float, Optional[str], float]:
    if not gallery:
        return None, 0.0, None, 0.0
    items = []
    for face_id, emb in gallery.items():
        emb_norm = _l2_normalize(emb)
        sim = _cosine(prototype, emb_norm)
        items.append((sim, face_id))
    items.sort(reverse=True, key=lambda x: (x[0], x[1]))
    top1_id, top1_sim = items[0][1], items[0][0]
    top2_id = None
    top2_sim = 0.0
    for sim, face_id in items[1:]:
        if face_id != top1_id:
            top2_id, top2_sim = face_id, sim
            break
    return top1_id, top1_sim, top2_id, top2_sim
