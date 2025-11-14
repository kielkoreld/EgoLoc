from __future__ import annotations

import math
from typing import Tuple

from .types import Position, Vehicle, ScoringParams


def _euclidean_distance(a: Position, b: Position) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)




def compute_enhanced_comm_efficiency(
    vehicle: Vehicle,
    reference_position: Position,
    params: ScoringParams,
    scene_complexity: float,  # 장면 복잡도 (0-1) - 필수 파라미터
    vehicle_density: float,   # 차량 밀도 (0-1) - 필수 파라미터
    neighbor_density: int = 0,
    communication_history: dict = None,  # 통신 이력
) -> Tuple[float, dict]:
    """
    향상된 통신 효율성 계산 - 다중 기준 및 통신 패턴 고려
    
    Args:
        vehicle: 평가할 차량
        reference_position: 기준 위치 (중심점)
        params: 스코어링 파라미터
        neighbor_density: 주변 차량 밀도
        scene_complexity: 장면 복잡도 (0-1)
        vehicle_density: 차량 밀도 (0-1)
        communication_history: 차량별 통신 이력
    
    Returns:
        (score, meta): 점수와 메타데이터
    """
    distance_m = _euclidean_distance(vehicle.position, reference_position)
    
    # 1. 거리 기반 기본 점수
    range_factor = max(0.0, 1.0 - (distance_m / max(1e-6, params.max_range_m)))
    
    # 2. 장면 복잡도 고려 (복잡할수록 낮은 점수)
    # 이미 정규화된 값이므로 직접 사용
    complexity_factor = max(0.0, 1.0 - scene_complexity)
    
    # 3. 차량 밀도 고려 (밀도가 높을수록 낮은 점수)
    # 이미 정규화된 값이므로 직접 사용
    density_factor = max(0.0, 1.0 - vehicle_density)
    
    # 4. 통신 이력 기반 점수 (과거 통신량이 적을수록 높은 점수)
    if communication_history and vehicle.vehicle_id in communication_history:
        historical_records = communication_history[vehicle.vehicle_id]
        if isinstance(historical_records, list) and len(historical_records) > 0:
            # 통신 기록에서 bytes_transmitted 추출
            usage_values = []
            for record in historical_records:
                if isinstance(record, dict) and 'bytes_transmitted' in record:
                    usage_values.append(record['bytes_transmitted'])
                elif isinstance(record, (int, float)):
                    # 기존 형식과의 호환성
                    usage_values.append(record)
            
            if usage_values:
                avg_usage = sum(usage_values) / len(usage_values)
                # 과거 사용량이 적을수록 높은 점수 (파라미터 기반 기준)
                usage_efficiency = 1.0 / (1.0 + avg_usage / params.usage_efficiency_threshold)
            else:
                usage_efficiency = 1.0
        else:
            usage_efficiency = 1.0
    else:
        usage_efficiency = 1.0  # 이력이 없으면 중립
    
    # 5. 통신 안정성 (분산이 적을수록 안정적)
    if communication_history and vehicle.vehicle_id in communication_history:
        historical_records = communication_history[vehicle.vehicle_id]
        if isinstance(historical_records, list) and len(historical_records) > 1:
            # 통신 기록에서 bytes_transmitted 추출
            usage_values = []
            for record in historical_records:
                if isinstance(record, dict) and 'bytes_transmitted' in record:
                    usage_values.append(record['bytes_transmitted'])
                elif isinstance(record, (int, float)):
                    # 기존 형식과의 호환성
                    usage_values.append(record)
            
            if len(usage_values) > 1:
                avg_usage = sum(usage_values) / len(usage_values)
                variance = sum((x - avg_usage)**2 for x in usage_values) / len(usage_values)
                # 분산이 적을수록 높은 점수 (파라미터 기반 기준)
                stability_factor = 1.0 / (1.0 + variance / (params.stability_threshold)**2)
            else:
                stability_factor = 1.0
        else:
            stability_factor = 1.0
    else:
        stability_factor = 1.0
    
    # 6. 혼잡도 페널티
    congestion_penalty = 1.0 / (1.0 + params.congestion_weight * max(0, neighbor_density))
    
    # 최종 점수 계산 (가중 평균) - 파라미터 기반 가중치
    raw_score = 100.0 * (
        params.range_weight * range_factor +           # 거리 기반
        params.complexity_weight * complexity_factor +  # 복잡도
        params.density_weight * density_factor +       # 밀도
        params.usage_efficiency_weight * usage_efficiency +  # 사용량 효율성
        params.stability_weight * stability_factor     # 안정성
    ) * congestion_penalty
    
    score = max(params.min_score_clip, min(params.max_score_clip, raw_score))
    
    meta = {
        "distance_m": distance_m,
        "range_factor": range_factor,
        "complexity_factor": complexity_factor,
        "density_factor": density_factor,
        "usage_efficiency": usage_efficiency,
        "stability_factor": stability_factor,
        "congestion_penalty": congestion_penalty,
        "raw_score": raw_score,
        "final_score": score,
    }
    
    return score, meta




