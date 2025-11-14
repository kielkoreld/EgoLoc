from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from ..types import (
    Position,
    Vehicle,
    CollaboratorObservation,
    DatasetSample,
)


def _position_from_dict(d: Dict[str, Any]) -> Position:
    return Position(
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        z=float(d.get("z", 0.0)),
        timestamp_s=d.get("timestamp_s"),
    )


def build_sample_from_dict(payload: Dict[str, Any]) -> DatasetSample:
    """기존 딕셔너리 형태의 데이터를 DatasetSample로 변환"""
    vehicles_src: List[Dict[str, Any]] = payload.get("vehicles", [])
    collab_src: List[Dict[str, Any]] = payload.get("collaborators", [])
    ref_src: Dict[str, Any] = payload.get("reference_position", {})

    vehicles: List[Vehicle] = []
    for v in vehicles_src:
        pos = _position_from_dict(v.get("position", {}))
        vehicles.append(
            Vehicle(
                vehicle_id=str(v.get("vehicle_id")),
                position=pos,
                heading_deg=v.get("heading_deg"),
                speed_mps=v.get("speed_mps"),
            )
        )

    collaborators: List[CollaboratorObservation] = []
    for c in collab_src:
        rel = c.get("relative_position")
        rel_pos = _position_from_dict(rel) if isinstance(rel, dict) else None
        collaborators.append(
            CollaboratorObservation(
                observer_id=str(c.get("observer_id")),
                target_vehicle_id=str(c.get("target_vehicle_id")),
                relative_position=rel_pos,
                signal_quality=float(c.get("signal_quality", 0.0)),
                trust_score=float(c.get("trust_score", 0.0)),
                timestamp_s=c.get("timestamp_s"),
            )
        )

    reference_position = _position_from_dict(ref_src)

    return DatasetSample(
        vehicles=vehicles,
        collaborators=collaborators,
        reference_position=reference_position,
    )


def build_sample_from_v2x_sim(
    data: Dict[str, Any], 
    dataset_path: Optional[Union[str, Path]] = None,
    frame_idx: int = 0
) -> DatasetSample:
    """
    V2X-Sim 데이터셋의 npy 파일에서 DatasetSample을 생성합니다.
    
    Args:
        data: V2X-Sim npy 파일에서 로드한 딕셔너리 데이터
        dataset_path: 데이터셋 경로 (선택사항, 메타데이터용)
        frame_idx: 프레임 인덱스 (선택사항)
    
    Returns:
        DatasetSample: VEVLoc 모듈에서 사용 가능한 형태로 변환된 데이터
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy가 필요합니다. pip install numpy로 설치하세요.")
    
    # 변환 행렬에서 차량 위치 추출
    trans_matrices = data.get("trans_matrices", np.array([]))
    target_agent_id = data.get("target_agent_id", 0)
    # 데이터셋에 존재하는 에이전트 수를 신뢰하여 동적으로 처리
    num_sensor = len(trans_matrices) if hasattr(trans_matrices, "__len__") else 0
    
    vehicles: List[Vehicle] = []
    collaborators: List[CollaboratorObservation] = []
    
    # 타겟(기준) 에이전트의 위치를 먼저 파악하여 상대 거리를 계산
    target_pos = None
    if 0 <= target_agent_id < len(trans_matrices):
        tm = trans_matrices[target_agent_id]
        if getattr(tm, "shape", None) == (4, 4):
            target_pos = (float(tm[0, 3]), float(tm[1, 3]), float(tm[2, 3]))

    # 각 에이전트(차량)의 위치 정보 추출
    for agent_idx in range(num_sensor):
        if agent_idx >= len(trans_matrices):
            break
            
        trans_matrix = trans_matrices[agent_idx]
        
        # 변환 행렬에서 위치 추출 (x, y, z)
        if trans_matrix.shape == (4, 4):
            x = float(trans_matrix[0, 3])
            y = float(trans_matrix[1, 3])
            z = float(trans_matrix[2, 3])
            
            # 회전 행렬에서 헤딩 각도 계산 (yaw)
            heading_rad = np.arctan2(trans_matrix[1, 0], trans_matrix[0, 0])
            heading_deg = float(np.degrees(heading_rad))
            
            vehicle_id = f"agent{agent_idx}"
            position = Position(x=x, y=y, z=z)
            
            vehicle = Vehicle(
                vehicle_id=vehicle_id,
                position=position,
                heading_deg=heading_deg,
                speed_mps=None  # V2X-Sim에서 속도 정보는 별도로 제공되지 않음
            )
            vehicles.append(vehicle)
            
            # 협력자 관측 정보 생성 (기준 에이전트와의 상대 거리 기반 신호 품질)
            if agent_idx != target_agent_id:
                # 기준 에이전트 기준 상대 좌표/거리 계산
                if target_pos is not None:
                    rx = x - target_pos[0]
                    ry = y - target_pos[1]
                    rz = z - target_pos[2]
                    distance = float(np.sqrt(rx*rx + ry*ry + rz*rz))
                    rel_position = Position(x=rx, y=ry, z=rz)
                else:
                    # 폴백: 절대 거리 사용
                    distance = float(np.sqrt(x*x + y*y + z*z))
                    rel_position = Position(x=x, y=y, z=z)
                signal_quality = max(0.1, min(1.0, 1.0 - distance / 200.0))  # 200m 기준
                trust_score = 0.8  # 기본 신뢰도
                
                collaborator = CollaboratorObservation(
                    observer_id=f"sensor_{agent_idx}",
                    target_vehicle_id=vehicle_id,
                    relative_position=rel_position,
                    signal_quality=signal_quality,
                    trust_score=trust_score,
                    timestamp_s=frame_idx * 0.1  # 가정: 10Hz
                )
                collaborators.append(collaborator)
    
    # 기준 위치는 타겟 에이전트의 위치로 설정
    reference_position = Position(x=0.0, y=0.0, z=0.0)
    if target_agent_id < len(vehicles):
        target_vehicle = vehicles[target_agent_id]
        reference_position = target_vehicle.position
    
    return DatasetSample(
        vehicles=vehicles,
        collaborators=collaborators,
        reference_position=reference_position,
    )


def load_v2x_sim_sample(
    dataset_path: Union[str, Path], 
    agent_id: int, 
    scenario_id: int, 
    frame_id: int
) -> DatasetSample:
    """
    V2X-Sim 데이터셋에서 특정 샘플을 로드합니다.
    
    Args:
        dataset_path: V2X-Sim 데이터셋 루트 경로
        agent_id: 에이전트 ID (예: 0, 1, 2, ...)
        scenario_id: 시나리오 ID (예: 0, 1, 2, ...)
        frame_id: 프레임 ID (예: 0, 1, 2, ...)
    
    Returns:
        DatasetSample: 변환된 데이터 샘플
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy가 필요합니다. pip install numpy로 설치하세요.")
    
    dataset_path = Path(dataset_path)
    agent_root = dataset_path / f"agent{agent_id}"
    npy_file = agent_root / f"{scenario_id}_{frame_id}" / "0.npy"

    if not npy_file.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {npy_file}")

    data = np.load(npy_file, allow_pickle=True).item()
    return build_sample_from_v2x_sim(data, dataset_path, frame_id)


def dataset_sample_to_dict(sample: DatasetSample) -> Dict[str, Any]:
    """DatasetSample을 JSON 직렬화 가능한 dict로 변환합니다.

    출력 스키마는 examples/sample_payload.json과 동일합니다.
    """
    vehicles = []
    for v in sample.vehicles:
        vehicles.append(
            {
                "vehicle_id": v.vehicle_id,
                "position": {"x": v.position.x, "y": v.position.y, "z": v.position.z, "timestamp_s": v.position.timestamp_s},
                "heading_deg": v.heading_deg,
                "speed_mps": v.speed_mps,
            }
        )

    collaborators = []
    for c in sample.collaborators:
        rel = None
        if c.relative_position is not None:
            rel = {
                "x": c.relative_position.x,
                "y": c.relative_position.y,
                "z": c.relative_position.z,
                "timestamp_s": c.relative_position.timestamp_s,
            }
        collaborators.append(
            {
                "observer_id": c.observer_id,
                "target_vehicle_id": c.target_vehicle_id,
                "relative_position": rel,
                "signal_quality": c.signal_quality,
                "trust_score": c.trust_score,
                "timestamp_s": c.timestamp_s,
            }
        )

    payload = {
        "reference_position": {
            "x": sample.reference_position.x,
            "y": sample.reference_position.y,
            "z": sample.reference_position.z,
            "timestamp_s": sample.reference_position.timestamp_s,
        },
        "vehicles": vehicles,
        "collaborators": collaborators,
    }
    return payload


def build_v2x_sim_payload(
    dataset_path: Union[str, Path],
    agent_id: int,
    scenario_id: int,
    frame_id: int,
) -> Dict[str, Any]:
    """V2X-Sim에서 단일 프레임을 로드하여 JSON(dict) 페이로드로 반환합니다."""
    sample = load_v2x_sim_sample(
        dataset_path=dataset_path,
        agent_id=agent_id,
        scenario_id=scenario_id,
        frame_id=frame_id,
    )
    return dataset_sample_to_dict(sample)


def save_v2x_sim_payload(
    out_file: Union[str, Path],
    dataset_path: Union[str, Path],
    agent_id: int,
    scenario_id: int,
    frame_id: int,
    ensure_dir: bool = True,
) -> Path:
    """V2X-Sim 단일 프레임을 JSON 파일로 저장합니다."""
    import json

    payload = build_v2x_sim_payload(
        dataset_path=dataset_path,
        agent_id=agent_id,
        scenario_id=scenario_id,
        frame_id=frame_id,
    )
    out_path = Path(out_file)
    if ensure_dir:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def list_v2x_sim_scenarios(
    dataset_path: Union[str, Path],
    agent_id: int,
) -> Dict[int, List[int]]:
    """에이전트 폴더를 스캔하여 {scenario_id: [frame_ids...]} 매핑을 반환합니다.

    지원 패턴(단일):
    - agent{agent_id}/{scenario_id}_{frame_id}/0.npy
    """
    root = Path(dataset_path) / f"agent{agent_id}"
    if not root.exists():
        return {}

    scenarios: Dict[int, List[int]] = {}

    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if "_" not in name:
            continue
        try:
            s_str, f_str = name.split("_", 1)
            s_id = int(s_str)
            f_id = int(f_str)
        except ValueError:
            continue
        zero_file = child / "0.npy"
        if zero_file.exists():
            scenarios.setdefault(s_id, []).append(f_id)

    # 프레임 ID 정렬 및 중복 제거
    for k in list(scenarios.keys()):
        scenarios[k] = sorted(set(scenarios[k]))
    return scenarios



