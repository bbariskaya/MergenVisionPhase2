"""Best-shot selection service for canonical video persons.

Best-shot selection is primarily performed inside reconciliation. This
standalone service is a placeholder for callers that receive a canonical
person without its raw observations and want to recompute or flag the best
shot once additional metadata is joined.
"""
from __future__ import annotations

from typing import Sequence, Tuple

from app.domain.video_tracking import CanonicalVideoPerson


class SelectBestShots:
    def __call__(
        self, persons: Sequence[CanonicalVideoPerson]
    ) -> Tuple[CanonicalVideoPerson, ...]:
        return tuple(persons)
