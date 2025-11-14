from __future__ import annotations

import heapq
import itertools
from typing import List, Optional

from .types import Candidate


class CandidatePriorityQueue:
    """점수 기반 우선순위 큐 (높은 점수 우선)."""

    def __init__(self) -> None:
        self._heap: List[tuple] = []
        self._counter = itertools.count()

    def push(self, candidate: Candidate) -> None:
        # heapq는 최소 힙이므로 음수 점수로 저장
        heapq.heappush(self._heap, (-candidate.score, next(self._counter), candidate))

    def pop(self) -> Optional[Candidate]:
        if not self._heap:
            return None
        _, _, cand = heapq.heappop(self._heap)
        return cand

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._heap)


