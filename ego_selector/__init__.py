"""
Ego Selector: 위치 기반 통신 효율 스코어링과 협력자 검증을 통해
후보 차량 중 자아 차량(ego vehicle)을 선택하는 프레임워크-프리 전처리 모듈.

핵심 구성요소
- types: 데이터 모델과 인터페이스
- scoring: 위치/링크 기반 통신 효율 스코어러
- priority_queue: 점수 기반 우선순위 큐
- validator: 협력자 교차검증 로직
- selector: 오케스트레이션 및 셀렉션 파이프라인
"""

from .types import (
    Position,
    Vehicle,
    Candidate,
    CollaboratorObservation,
    ValidationResult,
    SelectionResult,
    DatasetSample,
    ScoringParams,
)

from .scoring import compute_enhanced_comm_efficiency
from .priority_queue import CandidatePriorityQueue
from .selector import EgoSelector
from .validator import (
    prove_inclusion,
    ProofResult,
    peer_consensus_certification,
    PeerConsensus,
)
from .adapters import (
    build_sample_from_dict,
    build_sample_from_v2x_sim,
    load_v2x_sim_sample,
)

__all__ = [
    "Position",
    "Vehicle",
    "Candidate",
    "CollaboratorObservation",
    "ValidationResult",
    "SelectionResult",
    "DatasetSample",
    "ScoringParams",
    "compute_enhanced_comm_efficiency",
    "CandidatePriorityQueue",
    "EgoSelector",
    "prove_inclusion",
    "ProofResult",
    "peer_consensus_certification",
    "PeerConsensus",
    "build_sample_from_dict",
    "build_sample_from_v2x_sim",
    "load_v2x_sim_sample",
]


