"""Offline canonical person reconciliation from tracklet evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.domain.video_tracking import (
    AppearanceInterval,
    BestShotEvidence,
    CanonicalVideoPerson,
    RecognitionObservation,
    ReconciliationConfig,
    TrackletEvidence,
    _cosine,
    _is_valid_observation,
    _l2_normalize,
    _select_diverse_observations,
    _seconds_from_ns,
    build_tracklet_prototype,
    gallery_top_scores,
)


@dataclass
class _ResolvedTracklet:
    tracklet: TrackletEvidence
    prototype: Tuple[float, ...]
    face_id: Optional[str] = None
    status: str = "unknown"
    top1_similarity: float = 0.0
    similarity_margin: float = 0.0


@dataclass
class _Cluster:
    cluster_id: int
    members: Set[int] = field(default_factory=set)


class ReconcileVideoIdentities:
    def __init__(
        self,
        known_gallery: Dict[str, Sequence[float]],
        anonymous_gallery: Dict[str, Sequence[float]],
        config: ReconciliationConfig,
    ):
        self.known_gallery = known_gallery
        self.anonymous_gallery = anonymous_gallery
        self.cfg = config

    def __call__(self, tracklets: Sequence[TrackletEvidence]) -> Tuple[CanonicalVideoPerson, ...]:
        resolved = self._resolve_tracklets(tracklets)
        groups = self._group_tracklets(resolved)
        persons: list[CanonicalVideoPerson] = []
        for idx, group in enumerate(groups, start=1):
            person = self._build_person(f"vp_{idx:04d}", group)
            persons.append(person)
        return tuple(persons)

    def _resolve_tracklets(
        self, tracklets: Sequence[TrackletEvidence]
    ) -> Sequence[_ResolvedTracklet]:
        out: list[_ResolvedTracklet] = []
        for tl in tracklets:
            proto = build_tracklet_prototype(tl, self.cfg)
            if proto is None:
                # Keep unqualified tracklet as unknown with zero prototype so that
                # temporal/appearance metadata is still reported.
                proto = tuple(0.0 for _ in range(self.cfg.embedding_dim))
            rt = _ResolvedTracklet(tracklet=tl, prototype=proto)
            self._try_known(rt)
            if rt.status == "unknown":
                self._try_anonymous(rt)
            out.append(rt)
        return out

    def _try_known(self, rt: _ResolvedTracklet) -> None:
        top1_id, top1_sim, top2_id, top2_sim = gallery_top_scores(
            rt.prototype, self.known_gallery
        )
        margin = top1_sim - top2_sim if top2_id else 1.0
        if (
            top1_id is not None
            and top1_sim >= self.cfg.known_accept_top1_threshold
            and margin >= self.cfg.known_accept_margin_threshold
        ):
            rt.face_id = top1_id
            rt.status = "known"
            rt.top1_similarity = top1_sim
            rt.similarity_margin = margin

    def _try_anonymous(self, rt: _ResolvedTracklet) -> None:
        top1_id, top1_sim, top2_id, top2_sim = gallery_top_scores(
            rt.prototype, self.anonymous_gallery
        )
        margin = top1_sim - top2_sim if top2_id else 1.0
        if (
            top1_id is not None
            and top1_sim >= self.cfg.anonymous_match_top1_threshold
            and margin >= self.cfg.anonymous_match_margin_threshold
        ):
            rt.face_id = top1_id
            rt.status = "anonymous"
            rt.top1_similarity = top1_sim
            rt.similarity_margin = margin

    def _group_tracklets(
        self, resolved: Sequence[_ResolvedTracklet]
    ) -> Sequence[Sequence[_ResolvedTracklet]]:
        # Known and anonymous tracklets with same face_id group first.
        known_or_anon: Dict[Optional[str], List[int]] = {}
        unknown_indices: list[int] = []
        for i, rt in enumerate(resolved):
            if rt.status in ("known", "anonymous"):
                known_or_anon.setdefault(rt.face_id, []).append(i)
            else:
                unknown_indices.append(i)

        groups: list[Set[int]] = []
        # Deterministic ordering by face_id.
        for face_id in sorted(known_or_anon, key=lambda x: (x is None, x or "")):
            groups.append(set(known_or_anon[face_id]))

        if unknown_indices:
            unknown_clusters = self._cluster_unknowns(resolved, unknown_indices)
            groups.extend(unknown_clusters)

        # Build cannot-link-conflict-free groups. If hard constraints are violated,
        # split by moving conflicting tracks into separate singleton groups.
        final_groups: list[List[_ResolvedTracklet]] = []
        for g in groups:
            members = sorted(g, key=lambda idx: resolved[idx].tracklet.start_pts_ns)
            accepted: list[int] = []
            for idx in members:
                conflict = False
                for accepted_idx in accepted:
                    if self._cannot_link(resolved[idx], resolved[accepted_idx]):
                        conflict = True
                        break
                if conflict:
                    final_groups.append([resolved[idx]])
                else:
                    accepted.append(idx)
            if accepted:
                final_groups.append([resolved[i] for i in accepted])
        return final_groups

    def _cluster_unknowns(
        self, resolved: Sequence[_ResolvedTracklet], unknown_indices: Sequence[int]
    ) -> Sequence[Set[int]]:
        if not unknown_indices:
            return []

        labels = {idx: {idx} for idx in unknown_indices}
        cannot = set()
        for i, a in enumerate(unknown_indices):
            for b in unknown_indices[i + 1 :]:
                if self._cannot_link(resolved[a], resolved[b]):
                    cannot.add((a, b))

        while True:
            best_a: Optional[int] = None
            best_b: Optional[int] = None
            best_min_sim = -1.0
            # Deterministic candidate ordering.
            sorted_labels = sorted(labels)
            for i, label_a in enumerate(sorted_labels):
                for label_b in sorted_labels[i + 1 :]:
                    if (label_a, label_b) in cannot or (label_b, label_a) in cannot:
                        continue
                    if labels[label_a].isdisjoint(labels[label_b]):
                        members_a = labels[label_a]
                        members_b = labels[label_b]
                        # Complete-link: minimum pairwise similarity.
                        min_sim = 2.0
                        for ma in members_a:
                            for mb in members_b:
                                sim = _cosine(resolved[ma].prototype, resolved[mb].prototype)
                                if sim < min_sim:
                                    min_sim = sim
                        if min_sim > best_min_sim:
                            best_min_sim = min_sim
                            best_a, best_b = label_a, label_b

            if best_min_sim < self.cfg.unknown_cluster_threshold or best_a is None:
                break
            # Require margin over next best cluster? Simplistic: require complete-link sim.
            labels[best_a] = labels[best_a].union(labels[best_b])
            del labels[best_b]
            # Renormalize: every remaining label contains disjoint tracklet indices.

        # Return one set per discovered cluster.
        clusters: list[Set[int]] = []
        for label in sorted(labels):
            clusters.append(labels[label])
        return clusters

    @staticmethod
    def _cannot_link(a: _ResolvedTracklet, b: _ResolvedTracklet) -> bool:
        # Same source with overlapping time ranges.
        if (
            a.tracklet.source_id == b.tracklet.source_id
            and a.tracklet.start_pts_ns < b.tracklet.end_pts_ns
            and b.tracklet.start_pts_ns < a.tracklet.end_pts_ns
        ):
            return True
        # Strong conflicting known identities.
        if (
            a.status == "known"
            and b.status == "known"
            and a.face_id is not None
            and b.face_id is not None
            and a.face_id != b.face_id
        ):
            return True
        return False

    def _build_person(
        self, video_person_id: str, group: Sequence[_ResolvedTracklet]
    ) -> CanonicalVideoPerson:
        tracklets = [rt.tracklet for rt in group]
        tracklet_ids = tuple(t.tracklet_id for t in tracklets)

        # Determine canonical identity.
        face_id_counts: Dict[Optional[str], int] = {}
        status_counts: Dict[str, int] = {}
        total_top1 = 0.0
        total_margin = 0.0
        n_scored = 0
        for rt in group:
            face_id_counts[rt.face_id] = face_id_counts.get(rt.face_id, 0) + 1
            status_counts[rt.status] = status_counts.get(rt.status, 0) + 1
            if rt.status in ("known", "anonymous"):
                total_top1 += rt.top1_similarity
                total_margin += rt.similarity_margin
                n_scored += 1

        primary_face_id = max(face_id_counts, key=lambda k: (face_id_counts[k], k or ""))
        if status_counts.get("known", 0) > 0:
            status = "known"
            name = primary_face_id
        elif status_counts.get("anonymous", 0) > 0:
            status = "anonymous"
            name = None
        else:
            status = "new_anonymous"
            primary_face_id = f"new_anon_{video_person_id}"
            name = None

        if n_scored == 0:
            final_confidence = 0.0
        else:
            avg_top1 = total_top1 / n_scored
            avg_margin = total_margin / n_scored
            final_confidence = min(1.0, max(0.0, avg_top1 * (1.0 + avg_margin) / 2.0))

        appearances = self._build_appearances(group)
        total_duration = sum(iv.duration_ns() for iv in appearances)
        first_seen_ns = min(iv.start_pts_ns for iv in appearances)
        last_seen_ns = max(iv.end_pts_ns for iv in appearances)

        best_shot = self._select_best_shot(group)

        return CanonicalVideoPerson(
            video_person_id=video_person_id,
            face_id=primary_face_id,
            status=status,
            name=name,
            tracklet_ids=tracklet_ids,
            appearances=tuple(appearances),
            first_seen=_seconds_from_ns(first_seen_ns),
            last_seen=_seconds_from_ns(last_seen_ns),
            total_duration=_seconds_from_ns(total_duration),
            final_confidence=final_confidence,
            best_shot=best_shot,
        )

    def _build_appearances(
        self, group: Sequence[_ResolvedTracklet]
    ) -> Sequence[AppearanceInterval]:
        observations: list[RecognitionObservation] = []
        for rt in group:
            observations.extend(rt.tracklet.observations)
        if not observations:
            return []
        sorted_obs = sorted(observations, key=lambda o: o.pts_ns)
        intervals: list[AppearanceInterval] = []
        current: Optional[AppearanceInterval] = None
        current_source: Optional[int] = None
        current_tracklet: Optional[str] = None
        for o in sorted_obs:
            new_interval = False
            if current is None:
                new_interval = True
            elif current_source != self._find_source_for_observation(group, o):
                new_interval = True
            elif current_tracklet != self._find_tracklet_for_observation(group, o):
                new_interval = True
            elif (o.pts_ns - current.end_pts_ns) > self.cfg.appearance_gap_ns:
                new_interval = True

            if new_interval:
                if current is not None:
                    intervals.append(current)
                current = AppearanceInterval(start_pts_ns=o.pts_ns, end_pts_ns=o.pts_ns)
                current_source = self._find_source_for_observation(group, o)
                current_tracklet = self._find_tracklet_for_observation(group, o)
            else:
                current = AppearanceInterval(
                    start_pts_ns=current.start_pts_ns, end_pts_ns=o.pts_ns
                )
        if current is not None:
            intervals.append(current)
        return intervals

    @staticmethod
    def _find_source_for_observation(
        group: Sequence[_ResolvedTracklet], observation: RecognitionObservation
    ) -> int:
        for rt in group:
            for o in rt.tracklet.observations:
                if o.detection_id == observation.detection_id:
                    return rt.tracklet.source_id
        return -1

    @staticmethod
    def _find_tracklet_for_observation(
        group: Sequence[_ResolvedTracklet], observation: RecognitionObservation
    ) -> str:
        for rt in group:
            for o in rt.tracklet.observations:
                if o.detection_id == observation.detection_id:
                    return rt.tracklet.tracklet_id
        return ""

    @staticmethod
    def _select_best_shot(group: Sequence[_ResolvedTracklet]) -> BestShotEvidence:
        candidates: list[RecognitionObservation] = []
        for rt in group:
            valid = [o for o in rt.tracklet.observations if _is_valid_observation(o, ReconciliationConfig())]
            diverse = _select_diverse_observations(valid, ReconciliationConfig())
            candidates.extend(diverse)
        if not candidates:
            # Fallback: use first observation regardless of quality.
            for rt in group:
                if rt.tracklet.observations:
                    o = rt.tracklet.observations[0]
                    return BestShotEvidence(
                        frame=o.frame,
                        pts_ns=o.pts_ns,
                        detection_id=o.detection_id,
                        quality=0.0,
                        embedding_ref=None,
                    )
            return BestShotEvidence(frame=0, pts_ns=0, detection_id="", quality=0.0)

        def quality(o: RecognitionObservation) -> float:
            q = o.sharpness_score
            if o.frontal_score > 0.0:
                q += 0.3 * o.frontal_score
            if o.embedding_quality > 0.0:
                q += 0.2 * o.embedding_quality
            return q

        best = max(candidates, key=lambda o: (quality(o), -o.pts_ns))
        return BestShotEvidence(
            frame=best.frame,
            pts_ns=best.pts_ns,
            detection_id=best.detection_id,
            quality=quality(best),
            embedding_ref=None,
        )
