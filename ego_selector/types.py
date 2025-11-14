from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float = 0.0
    timestamp_s: Optional[float] = None


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: str
    position: Position
    heading_deg: Optional[float] = None
    speed_mps: Optional[float] = None


@dataclass
class Candidate:
    vehicle: Vehicle
    score: float
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CollaboratorObservation:
    observer_id: str
    target_vehicle_id: str
    relative_position: Optional[Position]
    signal_quality: float  # 0.0 ~ 1.0
    trust_score: float  # 0.0 ~ 1.0
    timestamp_s: Optional[float] = None


@dataclass(frozen=True)
class ValidationResult:
    target_vehicle_id: str
    is_valid: bool
    confidence: float
    reasons: List[str]


@dataclass(frozen=True)
class SelectionResult:
    selected_vehicle_id: Optional[str]
    candidate: Optional[Candidate]
    validation: Optional[ValidationResult]
    audited_count: int


@dataclass(frozen=True)
class DatasetSample:
    vehicles: List[Vehicle]
    collaborators: List[CollaboratorObservation]
    reference_position: Position


@dataclass(frozen=True)
class CommunicationRecord:
    """통신 기록"""
    timestamp: float
    bytes_transmitted: int
    success: bool
    latency_ms: float = 0.0


@dataclass(frozen=True)
class VehicleCommunicationHistory:
    """차량별 통신 이력"""
    vehicle_id: str
    records: List[CommunicationRecord]
    
    def get_recent_usage(self, window_minutes: int = 10) -> List[int]:
        """최근 window_minutes 동안의 통신량 반환"""
        current_time = max(r.timestamp for r in self.records) if self.records else 0
        cutoff_time = current_time - (window_minutes * 60)
        return [r.bytes_transmitted for r in self.records if r.timestamp >= cutoff_time]
    
    def get_usage_variance(self, window_minutes: int = 10) -> float:
        """최근 window_minutes 동안의 통신량 분산 반환"""
        recent_usage = self.get_recent_usage(window_minutes)
        if len(recent_usage) <= 1:
            return 0.0
        mean_usage = sum(recent_usage) / len(recent_usage)
        variance = sum((x - mean_usage) ** 2 for x in recent_usage) / len(recent_usage)
        return variance


@dataclass(frozen=True)
class ScoringParams:
    # 기본 통신 파라미터
    max_range_m: float = 250.0                    # 최대 통신 범위 (m)
    congestion_weight: float = 0.1               # 혼잡도 가중치
    min_score_clip: float = 0.0                   # 최소 점수 클리핑
    max_score_clip: float = 100.0                 # 최대 점수 클리핑
    
    # 통신 이력 기준값 파라미터
    usage_efficiency_threshold: float = 100.0 * 1024 * 1024  # 100MB 기준
    stability_threshold: float = 50.0 * 1024 * 1024           # 50MB 기준
    
    # 최종 점수 가중치 파라미터
    range_weight: float = 0.35           # 거리 기반 가중치 (35%)
    complexity_weight: float = 0.20       # 복잡도 가중치 (20%)
    density_weight: float = 0.20          # 밀도 가중치 (20%)
    usage_efficiency_weight: float = 0.15  # 사용량 효율성 가중치 (15%)
    stability_weight: float = 0.10        # 안정성 가중치 (10%)


@dataclass(frozen=True)
class SelectionResult:
    """자아 차량 선택 결과"""
    selected_vehicle_id: Optional[str]
    candidate: Optional[Candidate]
    validation: Optional[ValidationResult]
    audited_count: int
    queue_order: Optional[Dict[str, float]] = None  # 큐 순서 정보


