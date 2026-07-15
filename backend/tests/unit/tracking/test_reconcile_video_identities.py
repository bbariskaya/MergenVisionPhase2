"""Unit tests for Sprint 06 offline canonical identity reconciliation."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import pytest

from app.domain.video_tracking import (
    RecognitionObservation,
    ReconciliationConfig,
    TrackletEvidence,
)
from app.application.services.reconcile_video_identities import (
    ReconcileVideoIdentities,
)


EMBEDDING_DIM = 512


def _normalized(base: float, dim: int = EMBEDDING_DIM) -> Tuple[float, ...]:
    vec = [base + i * 0.0001 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return tuple(v / norm for v in vec)


def _identity_emb(index: int, dim: int = EMBEDDING_DIM) -> Tuple[float, ...]:
    vec = [0.0001 * (i + 1) for i in range(dim)]
    vec[index] = 1.0
    return _normalized_vector(vec)


def _mixed_emb(indices: Sequence[int], dim: int = EMBEDDING_DIM) -> Tuple[float, ...]:
    vec = [0.0001 * (i + 1) for i in range(dim)]
    for idx in indices:
        vec[idx] = 1.0
    return _normalized_vector(vec)


def _normalized_vector(vec: Sequence[float]) -> Tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in vec))
    return tuple(v / norm for v in vec)


def _alice_emb() -> Tuple[float, ...]:
    return _identity_emb(0)


def _bob_emb() -> Tuple[float, ...]:
    return _identity_emb(1)


def _obs(
    detection_id: str,
    frame: int,
    pts_ns: int,
    embedding: Tuple[float, ...],
    face_size: float = 64.0,
    sharpness: float = 0.9,
    frontal: float = 0.8,
    quality: float = 0.8,
) -> RecognitionObservation:
    return RecognitionObservation(
        detection_id=detection_id,
        frame=frame,
        pts_ns=pts_ns,
        bbox=(10.0, 10.0, 74.0, 74.0),
        detector_score=0.85,
        embedding=embedding,
        embedding_quality=quality,
        face_width=face_size,
        face_height=face_size,
        border_clip_fraction=0.0,
        alignment_valid=True,
        frontal_score=frontal,
        sharpness_score=sharpness,
    )


def _tracklet(
    tracklet_id: str,
    source_id: int,
    start_pts_ns: int,
    end_pts_ns: int,
    embeddings: Sequence[Tuple[float, ...]],
) -> TrackletEvidence:
    start_frame = int(start_pts_ns / 33_333_333) + 1
    end_frame = int(end_pts_ns / 33_333_333) + 1
    if len(embeddings) == 0:
        raise ValueError("At least one embedding required")
    steps = max(len(embeddings), 2)
    span_ns = end_pts_ns - start_pts_ns
    step_ns = span_ns // (steps - 1) if steps > 1 else 0
    obs = []
    for i in range(steps):
        pts = start_pts_ns + step_ns * i
        frame = start_frame + i
        emb = embeddings[i % len(embeddings)]
        obs.append(_obs(f"{tracklet_id}_d{i}", frame, pts, emb))
    return TrackletEvidence(
        tracklet_id=tracklet_id,
        source_id=source_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_pts_ns=start_pts_ns,
        end_pts_ns=end_pts_ns,
        observations=tuple(obs),
    )


@pytest.fixture
def config() -> ReconciliationConfig:
    return ReconciliationConfig(
        known_accept_top1_threshold=0.50,
        known_accept_margin_threshold=0.10,
        anonymous_match_top1_threshold=0.45,
        unknown_cluster_threshold=0.50,
    )


def test_same_known_person_distant_tracklets_merge(config):
    gallery = {"face_alice": _alice_emb()}
    reconciler = ReconcileVideoIdentities(gallery, {}, config)

    tl1 = _tracklet("s0_tl_0007", 0, 1_000_000_000, 2_000_000_000, [_alice_emb()])
    tl2 = _tracklet("s0_tl_0191", 0, 7_000_000_000, 7_500_000_000, [_alice_emb()])

    persons = reconciler([tl1, tl2])
    assert len(persons) == 1
    p = persons[0]
    assert p.status == "known"
    assert p.face_id == "face_alice"
    assert set(p.tracklet_ids) == {"s0_tl_0007", "s0_tl_0191"}
    assert pytest.approx(p.first_seen, abs=0.1) == 1.0
    assert pytest.approx(p.last_seen, abs=0.1) == 7.5
    assert p.total_duration > 0


def test_two_simultaneous_known_faces_do_not_merge(config):
    gallery = {"face_alice": _alice_emb(), "face_bob": _bob_emb()}
    reconciler = ReconcileVideoIdentities(gallery, {}, config)

    tl1 = _tracklet("s0_tl_0001", 0, 1_000_000_000, 2_000_000_000, [_alice_emb()])
    tl2 = _tracklet("s0_tl_0002", 0, 1_200_000_000, 2_200_000_000, [_bob_emb()])

    persons = reconciler([tl1, tl2])
    assert len(persons) == 2
    face_ids = {p.face_id for p in persons}
    assert face_ids == {"face_alice", "face_bob"}


def test_unknown_tracklets_above_threshold_merge(config):
    emb = _normalized(0.3)
    reconciler = ReconcileVideoIdentities({}, {}, config)

    tl1 = _tracklet("s0_tl_0010", 0, 1_000_000_000, 2_000_000_000, [emb])
    tl2 = _tracklet("s0_tl_0011", 0, 5_000_000_000, 6_000_000_000, [emb])

    persons = reconciler([tl1, tl2])
    assert len(persons) == 1
    assert persons[0].status == "new_anonymous"
    assert set(persons[0].tracklet_ids) == {"s0_tl_0010", "s0_tl_0011"}


def test_complete_link_chain_no_three_way_merge(config):
    # A-B similar, B-C similar, A-C dissimilar -> no single cluster.
    emb_a = _identity_emb(0)
    emb_b = _mixed_emb([0, 1])
    emb_c = _identity_emb(1)
    reconciler = ReconcileVideoIdentities({}, {}, config)

    tl_a = _tracklet("s0_tl_a", 0, 1_000_000_000, 2_000_000_000, [emb_a])
    tl_b = _tracklet("s0_tl_b", 0, 3_000_000_000, 4_000_000_000, [emb_b])
    tl_c = _tracklet("s0_tl_c", 0, 5_000_000_000, 6_000_000_000, [emb_c])

    persons = reconciler([tl_a, tl_b, tl_c])
    # Complete-link clustering: A may merge with B, B may merge with C if
    # individually similar, but A and C are dissimilar so they cannot all end
    # up in one cluster.
    all_tracklets = set()
    for p in persons:
        all_tracklets.update(p.tracklet_ids)
    assert all_tracklets == {"s0_tl_a", "s0_tl_b", "s0_tl_c"}
    assert not any({"s0_tl_a", "s0_tl_b", "s0_tl_c"}.issubset(set(p.tracklet_ids)) for p in persons)


def test_appearances_separated_by_gap(config):
    emb = _alice_emb()
    gallery: Dict[str, Tuple[float, ...]] = {"face_alice": emb}
    reconciler = ReconcileVideoIdentities(gallery, {}, config)

    # Single tracklet with a large internal gap.
    pts = [0, 33_333_333, 66_666_666, 5_000_000_000, 5_033_333_333]
    obs = tuple(_obs(f"f{i}", i, pts[i], emb) for i in range(len(pts)))
    tl = TrackletEvidence(
        tracklet_id="s0_tl_gap",
        source_id=0,
        start_frame=0,
        end_frame=150,
        start_pts_ns=pts[0],
        end_pts_ns=pts[-1],
        observations=obs,
    )
    persons = reconciler([tl])
    assert len(persons[0].appearances) == 2


def test_first_last_and_total_duration_exact(config):
    emb = _alice_emb()
    gallery = {"face_alice": emb}
    reconciler = ReconcileVideoIdentities(gallery, {}, config)

    tl = _tracklet("s0_tl_exact", 0, 1_000_000_000, 2_000_000_000, [emb])
    persons = reconciler([tl])
    p = persons[0]
    assert pytest.approx(p.first_seen, abs=0.001) == 1.0
    assert pytest.approx(p.last_seen, abs=0.001) == 2.0
    assert pytest.approx(p.total_duration, abs=0.001) == 1.0


def test_existing_anonymous_match(config):
    emb = _normalized(0.4)
    anonymous_gallery = {"anon_old_1": emb}
    reconciler = ReconcileVideoIdentities({}, anonymous_gallery, config)
    tl = _tracklet("s0_tl_old", 0, 1_000_000_000, 2_000_000_000, [emb])
    persons = reconciler([tl])
    assert len(persons) == 1
    assert persons[0].status == "anonymous"
    assert persons[0].face_id == "anon_old_1"
