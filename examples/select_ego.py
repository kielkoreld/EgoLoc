from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 모듈 검색 경로에 추가하여 어디서 실행하더라도 import가 동작하도록 함
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego_selector import EgoSelector, ScoringParams, build_sample_from_dict, compute_enhanced_comm_efficiency
from ego_selector import prove_inclusion, peer_consensus_certification
from ego_selector.adapters import build_v2x_sim_payload, save_v2x_sim_payload
import io as _io
import contextlib as _ctx
from pathlib import Path as _P
import types as _types
import os as _os
import time
import csv as _csv
import json as _json
import importlib as _importlib
import subprocess as _subprocess

def time_str() -> str:
    """현재 시간을 문자열로 반환 (YYYYMMDD_HHMMSS 형식)"""
    t = time.time()
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t))


def _clear_cuda_cache():
    """PyTorch CUDA 캐시와 Python GC를 정리"""
    try:
        import torch
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception:
        pass


def _normalize_robosac_path(path, project_root: Path) -> Path:
    """
    ROBOSAC 경로를 정규화하여 프로젝트 루트 기준 올바른 경로 반환.
    
    Args:
        path: 정규화할 경로 (Path 객체 또는 문자열)
        project_root: 프로젝트 루트 경로
        
    Returns:
        정규화된 Path 객체 (프로젝트 루트 기준)
    """
    if path is None:
        return (project_root / "ROBOSAC").resolve()
    
    path_obj = _P(path).resolve()
    
    # 경로가 프로젝트 루트 하위에 있으면 그대로 사용
    try:
        path_obj.relative_to(project_root)
        return path_obj
    except ValueError:
        # 프로젝트 루트 하위가 아니면 기본 경로 사용
        return (project_root / "ROBOSAC").resolve()


def _ensure_correct_path_in_sys_path(correct_path: Path, path_description: str = ""):
    """
    sys.path에 올바른 경로가 포함되어 있는지 확인하고 추가.
    이미 포함되어 있으면 중복 추가하지 않음.
    
    Args:
        correct_path: 추가할 경로
        path_description: 디버깅용 경로 설명
    """
    correct_path_str = str(correct_path.resolve())
    if correct_path_str not in sys.path:
        sys.path.insert(0, correct_path_str)
        if path_description:
            print(f"[ROBOSAC] {path_description} 경로 추가: {correct_path_str}")


def _cleanup_incorrect_modules_in_sys_modules(correct_root: Path, module_patterns: list[str] = None):
    """
    sys.modules에서 잘못된 경로의 모듈을 제거.
    특정 경로가 아닌, 프로젝트 루트 하위가 아닌 모듈을 찾아 제거.
    
    Args:
        correct_root: 올바른 루트 경로 (PROJECT_ROOT / "ROBOSAC")
        module_patterns: 제거할 모듈 이름 패턴 리스트 (예: ['coperception', 'robosac'])
    """
    if module_patterns is None:
        module_patterns = ['coperception', 'robosac']
    
    modules_to_remove = []
    correct_root_str = str(correct_root.resolve())
    
    for module_name, module_obj in list(sys.modules.items()):
        if module_obj is None:
            continue
            
        module_file = getattr(module_obj, '__file__', None)
        if not module_file:
            continue
        
        module_file_str = str(module_file)
        
        # 모듈 이름이 패턴에 맞고, 경로가 올바른 루트 하위가 아니면 제거 대상
        should_check = any(pattern in module_name for pattern in module_patterns)
        if should_check and not module_file_str.startswith(correct_root_str):
            modules_to_remove.append((module_name, module_file_str))
    
    # 모듈 제거 (하위 모듈부터 먼저 제거하여 의존성 문제 방지)
    # 모듈 이름을 정렬하여 더 긴 이름(하위 모듈)을 먼저 제거
    modules_to_remove.sort(key=lambda x: len(x[0]), reverse=True)
    
    for module_name, module_file in modules_to_remove:
        try:
            del sys.modules[module_name]
            print(f"[ROBOSAC] 잘못된 경로의 모듈 제거: {module_name} at {module_file}")
        except KeyError:
            pass  # 이미 제거된 경우
    
    return len(modules_to_remove)


def _load_robosac_module(robocfg: dict, project_root: Path):
    """
    ROBOSAC 모듈을 로드하고 경로를 정리하는 범용 함수.
    
    Args:
        robocfg: ROBOSAC 설정 딕셔너리 (root_path 포함)
        project_root: 프로젝트 루트 경로
        
    Returns:
        로드된 robosac 모듈 또는 None
    """
    # 1. 경로 정규화
    robosac_root = _normalize_robosac_path(robocfg.get("root_path"), project_root)
    robosac_root_str = str(robosac_root.resolve())
    
    # 경로 존재 확인
    if not robosac_root.exists():
        print(f"[ROBOSAC] 경고: ROBOSAC 루트 경로가 존재하지 않음: {robosac_root_str}")
        return None
    
    # 2. sys.path에 올바른 경로 추가 (기존 ROBOSAC 관련 경로는 유지)
    # coperception 패키지 경로 추가 (중요: 패키지 구조를 인식하기 위해)
    coperception_path = robosac_root / "coperception"
    if coperception_path.exists():
        _ensure_correct_path_in_sys_path(coperception_path, "coperception 패키지")
    _ensure_correct_path_in_sys_path(robosac_root, "ROBOSAC 루트")
    det_dir = robosac_root / "coperception" / "tools" / "det"
    if det_dir.exists():
        _ensure_correct_path_in_sys_path(det_dir, "ROBOSAC det")
    
    # 3. 모듈 import 시도 (여러 방법 시도)
    last_error = None
    try:
        from coperception.tools.det import robosac as robosac_module
        print(f"[ROBOSAC] coperception.tools.det에서 import 성공")
        return robosac_module
    except ImportError as e:
        last_error = e
        try:
            import robosac as robosac_module
            print(f"[ROBOSAC] 직접 robosac import 성공")
            return robosac_module
        except ImportError as e2:
            last_error = e2
            try:
                from tools.det import robosac as robosac_module
                print(f"[ROBOSAC] tools.det에서 import 성공")
                return robosac_module
            except ImportError as e3:
                last_error = e3
    
    # 모든 import 시도 실패 시 상세 오류 정보 출력
    print(f"[ROBOSAC] 모든 import 방법 실패")
    print(f"[ROBOSAC] ROBOSAC 루트 경로: {robosac_root_str}")
    print(f"[ROBOSAC] coperception 경로 존재: {coperception_path.exists() if coperception_path else False}")
    print(f"[ROBOSAC] det 디렉토리 존재: {det_dir.exists() if det_dir else False}")
    if det_dir.exists():
        robosac_file = det_dir / "robosac.py"
        print(f"[ROBOSAC] robosac.py 존재: {robosac_file.exists()}")
    print(f"[ROBOSAC] 마지막 오류: {last_error}")
    print(f"[ROBOSAC] sys.path 앞부분: {sys.path[:10]}")
    return None


def _reload_robosac_modules_if_needed(robosac_module, robocfg: dict, project_root: Path):
    """
    필요한 경우 ROBOSAC 관련 모듈을 reload.
    잘못된 경로의 모듈이 있으면 제거하고 올바른 경로에서 다시 로드.
    
    Args:
        robosac_module: 현재 로드된 robosac 모듈
        robocfg: ROBOSAC 설정 딕셔너리
        project_root: 프로젝트 루트 경로
        
    Returns:
        reload된 robosac 모듈
    """
    if robosac_module is None:
        return None
    
    # 경로 정규화
    robosac_root = _normalize_robosac_path(robocfg.get("root_path"), project_root)
    robosac_root_str = str(robosac_root.resolve())
    
    # 1단계: sys.path에서 ROBOSAC 관련 경로 모두 제거 (올바른 것 포함)
    sys.path = [p for p in sys.path if not any(
        pattern in str(p) for pattern in ['coperception', 'robosac', 'ROBOSAC']
    )]
    
    # 2단계: 잘못된 경로의 모든 모듈 제거 (sys.path 정리 후 수행)
    removed_count = _cleanup_incorrect_modules_in_sys_modules(robosac_root)
    if removed_count > 0:
        print(f"[ROBOSAC] {removed_count}개의 잘못된 경로 모듈 제거됨")
    
    # 3단계: coperception 패키지 전체를 sys.modules에서 제거 (하위 모듈 포함)
    # 하위 모듈부터 먼저 제거하여 의존성 문제 방지
    coperception_modules = [name for name in sys.modules.keys() if name.startswith('coperception')]
    coperception_modules.sort(key=len, reverse=True)  # 긴 이름(하위 모듈) 먼저
    for module_name in coperception_modules:
        try:
            del sys.modules[module_name]
        except KeyError:
            pass
    
    # 4단계: 올바른 경로를 sys.path 맨 앞에 추가 (우선순위 보장)
    coperception_path = robosac_root / "coperception"
    if coperception_path.exists():
        _ensure_correct_path_in_sys_path(coperception_path, "coperception 패키지 (reload)")
    _ensure_correct_path_in_sys_path(robosac_root, "ROBOSAC 루트 (reload)")
    det_dir = robosac_root / "coperception" / "tools" / "det"
    if det_dir.exists():
        _ensure_correct_path_in_sys_path(det_dir, "ROBOSAC det (reload)")
    
    # 5단계: CoDetModule 다시 import (이제 올바른 경로에서 로드됨)
    try:
        import coperception.utils.CoDetModule
        module_file = getattr(coperception.utils.CoDetModule, '__file__', None)
        if module_file:
            module_file_str = str(module_file)
            if not module_file_str.startswith(robosac_root_str):
                print(f"[ROBOSAC] 경고: CoDetModule이 여전히 잘못된 경로에서 로드됨: {module_file_str}")
                print(f"[ROBOSAC] 현재 sys.path 앞부분: {sys.path[:5]}")
            else:
                print(f"[ROBOSAC] CoDetModule reload 완료: {module_file_str}")
        else:
            print(f"[ROBOSAC] CoDetModule reload 완료 (__file__ 없음)")
    except Exception as e:
        print(f"[ROBOSAC] CoDetModule reload 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 6단계: robosac 모듈 reload
    try:
        robosac_module_file = getattr(robosac_module, '__file__', None)
        if robosac_module_file and not str(robosac_module_file).startswith(robosac_root_str):
            # 잘못된 경로면 sys.modules에서 제거하고 다시 import
            robosac_module_name = robosac_module.__name__
            robosac_related_modules = [
                name for name in sys.modules.keys() 
                if name == robosac_module_name or name.startswith(f"{robosac_module_name}.")
            ]
            for name in robosac_related_modules:
                try:
                    del sys.modules[name]
                except KeyError:
                    pass
            from coperception.tools.det import robosac
            robosac_module = robosac
            new_file = getattr(robosac_module, '__file__', None)
            if new_file and not str(new_file).startswith(robosac_root_str):
                print(f"[ROBOSAC] 경고: robosac 모듈이 여전히 잘못된 경로에서 로드됨: {new_file}")
            else:
                print(f"[ROBOSAC] robosac 모듈 다시 import 완료: {new_file}")
        else:
            _importlib.reload(robosac_module)
            reloaded_file = getattr(robosac_module, '__file__', None)
            if reloaded_file and not str(reloaded_file).startswith(robosac_root_str):
                print(f"[ROBOSAC] 경고: robosac 모듈 reload 후에도 잘못된 경로: {reloaded_file}")
            else:
                print(f"[ROBOSAC] robosac 모듈 reload 완료: {reloaded_file}")
    except Exception as e:
        print(f"[ROBOSAC] robosac 모듈 reload 실패: {e}")
        import traceback
        traceback.print_exc()
    
    return robosac_module

def _parse_agent_index(vid: str) -> int | None:
    """agent ID 문자열을 정수 인덱스로 파싱 (예: "agent3" -> 3)"""
    if isinstance(vid, str) and vid.startswith("agent"):
        try:
            return int(vid.replace("agent", ""))
        except Exception:
            return None
    return None


def _build_robosac_args_base(robocfg: dict, scenario_id: int, attack_mode: str | None = None) -> _types.SimpleNamespace:
    """ROBOSAC args 객체의 공통 기본값 설정"""
    args = _types.SimpleNamespace()
    args.data = robocfg.get("data_path")
    args.batch = 1
    args.nepoch = 100
    args.nworker = 4
    args.lr = 0.001
    args.log = False
    args.logpath = ""
    args.resume = robocfg.get("resume")
    args.resume_teacher = ""
    args.layer = 3
    args.warp_flag = False
    args.kd_flag = 0
    args.kd_weight = 100000
    args.gnn_iter_times = 3
    args.visualization = False
    args.com = robocfg.get("com", "mean")
    args.bound = "both"
    args.inference = None
    args.tracking = False
    args.box_com = False
    args.no_cross_road = False
    args.num_agent = robocfg.get("num_agent", 6)
    args.apply_late_fusion = 0
    args.compress_level = 0
    args.pose_noise = 0.0
    args.only_v2i = 0
    args.pert_alpha = 0.1
    args.adv_method = "pgd"
    # 공격 강도 매핑
    if attack_mode == "subtle":
        args.eps = 0.2
    elif attack_mode == "adaptive":
        args.eps = 0.8
    else:
        args.eps = 0.5
    args.adv_iter = robocfg.get("adv_iter", 15)
    args.scene_id = robocfg.get("scene_id", [scenario_id])
    args.sample_id = None
    args.robosac = robocfg.get("mode", "robosac_validation")
    args.robosac_k = robocfg.get("robosac_k", 3)
    args.ego_loss_only = False
    args.step_budget = robocfg.get("step_budget", 3)
    args.box_matching_thresh = 0.3
    args.number_of_attackers = robocfg.get("number_of_attackers", 1)
    args.fix_attackers = False
    args.use_history_frame = False
    args.partial_upperbound = False
    return args


def _serialize_selection_result(result) -> dict:
    """SelectionResult를 JSON 직렬화 가능한 dict로 변환 (numpy 스칼라 호환)."""
    def to_py(x):
        try:
            # numpy 스칼라(np.bool_, np.float64 등)는 item() 보유
            return x.item()  # type: ignore[attr-defined]
        except Exception:
            return x

    selected_id = result.selected_vehicle_id
    audited = int(result.audited_count) if result.audited_count is not None else None
    if result.validation:
        is_valid = result.validation.is_valid
        confidence = result.validation.confidence
        reasons = result.validation.reasons
        val_dict = {
            "is_valid": bool(to_py(is_valid)) if is_valid is not None else None,
            "confidence": float(to_py(confidence)) if confidence is not None else None,
            "reasons": reasons,
        }
    else:
        val_dict = None

    return {
        "selected_vehicle_id": selected_id,
        "audited_count": audited,
        "validation": val_dict,
    }


def _build_candidate_queue_dict(sample, scoring_params: ScoringParams) -> list[dict]:
    """선택 로직과 동일한 필터링/보정으로 후보 큐를 계산해 반환."""

    def _euclidean_distance_xyz(a, b) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _compute_reliability(target_vehicle_id: str) -> float:
        relevant = [o for o in sample.collaborators if o.target_vehicle_id == target_vehicle_id]
        if not relevant:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for obs in relevant:
            trust = max(0.0, min(1.0, obs.trust_score))
            quality = max(0.0, min(1.0, obs.signal_quality))
            weighted_sum += trust * quality
            weight_total += trust
        return 0.0 if weight_total == 0.0 else (weighted_sum / weight_total)

    def _compute_view_diversity(idx: int, radius_m: float = 60.0) -> float:
        me = sample.vehicles[idx]
        headings = []
        nearby = 0
        my_heading = me.heading_deg if me.heading_deg is not None else 0.0
        for j, other in enumerate(sample.vehicles):
            if j == idx:
                continue
            d = _euclidean_distance_xyz(me.position, other.position)
            if d <= radius_m:
                nearby += 1
                other_h = other.heading_deg if other.heading_deg is not None else my_heading
                diff = abs(((other_h - my_heading + 180.0) % 360.0) - 180.0)
                headings.append(diff / 180.0)
        if not headings:
            return 0.0
        spread = sum(headings) / len(headings)
        prox = min(nearby / 8.0, 1.0)
        val = 0.7 * spread + 0.3 * prox
        return max(0.0, min(1.0, val))

    neighbor_density = max(0, len(sample.vehicles) - 1)
    eps = getattr(scoring_params, "epsilon_m", 0.0)
    items = []
    for i, v in enumerate(sample.vehicles):
        # selector와 동일하게 참조점과 거의 같은 차량은 제외
        d_ref = _euclidean_distance_xyz(v.position, sample.reference_position)
        if d_ref <= max(1e-3, eps):
            continue
        # 향상된 스코어링 함수 사용 (selector와 동일)
        from ego_selector.selector import EgoSelector
        temp_selector = EgoSelector(scoring_params=scoring_params)
        
        # 장면 복잡도 및 차량 밀도 계산
        scene_complexity = temp_selector._compute_scene_complexity(sample)
        vehicle_density = temp_selector._compute_vehicle_density(sample)
        
        base_score, meta = compute_enhanced_comm_efficiency(
            vehicle=v,
            reference_position=sample.reference_position,
            params=scoring_params,
            scene_complexity=scene_complexity,
            vehicle_density=vehicle_density,
            neighbor_density=neighbor_density,
            communication_history={},  # 통신 이력 없음
        )
        reliability = _compute_reliability(v.vehicle_id)
        view_div = _compute_view_diversity(i)
        adj = (0.7 + 0.3 * reliability) * (0.85 + 0.15 * view_div)
        adjusted_score = max(scoring_params.min_score_clip, min(scoring_params.max_score_clip, base_score * adj))
        meta = dict(meta)
        meta.update({
            "reliability": reliability,
            "view_diversity": view_div,
            "adjust_multiplier": adj,
            "base_score": base_score,
        })
        items.append({
            "vehicle_id": v.vehicle_id,
            "score": float(adjusted_score),
            "metadata": {k: float(v) if isinstance(v, (int, float)) else v for k, v in meta.items()},
        })
    items.sort(key=lambda x: x["score"], reverse=True)
    for idx, it in enumerate(items, start=1):
        it["rank"] = idx
    return items


# 기존 대역폭 추정 함수들 제거됨 - 실제 측정만 사용


def _format_candidate_queue_line(sample, scoring_params: ScoringParams, vehicle_scores: dict = None) -> str:
    """후보 큐를 "id(score) -> id(score)" 단일 라인 문자열로 포맷."""
    if vehicle_scores is not None:
        # 저장된 점수 사용
        items = [{"vehicle_id": vid, "score": score} for vid, score in vehicle_scores.items()]
        items.sort(key=lambda x: x["score"], reverse=True)
    else:
        # 기존 방식 (점수 재계산)
        items = _build_candidate_queue_dict(sample, scoring_params)
    
    parts = [f"{it['vehicle_id']}({it['score']:.2f})" for it in items]
    return " -> ".join(parts)

def run_sample_example() -> None:
    """기본 샘플 JSON 파일을 사용한 예제"""
    payload_path = Path(__file__).parent / "sample_payload.json"
    data = json.loads(payload_path.read_text(encoding="utf-8"))

    sample = build_sample_from_dict(data)

    selector = EgoSelector(
        scoring_params=ScoringParams(
            max_range_m=200.0,  # 250.0 → 200.0 (더 엄격한 선택)
            congestion_weight=0.2,  # 0.1 → 0.2 (혼잡도 고려 강화)
        ),
        min_validation_confidence=0.6,
        attack_mode=None,  # 공격 모드 전달
    )

    result = selector.select(sample)
    print(json.dumps(_serialize_selection_result(result), ensure_ascii=False, indent=2))
    # 후보 큐 추가 출력 (간단 형식)
    queue_line = _format_candidate_queue_line(sample, scoring_params=ScoringParams(max_range_m=250.0))
    print(queue_line)


def run_v2x_sim_example(
    dataset_path: str,
    agent_id: int,
    scenario_id: int,
    frame_id: int,
    export_json_path: str | None = None,
    reference_mode: str = "center",
    save_logs: bool = False,
    attack_mode: str | None = None,
    log_bandwidth: bool = False,
    no_defense: bool = False,
) -> None:
    """V2X-Sim 데이터셋을 JSON으로 추출 후, 그 JSON을 기반으로 선택을 수행"""
    # 1) V2X-Sim에서 JSON 추출
    payload = build_v2x_sim_payload(
        dataset_path=dataset_path,
        agent_id=agent_id,
        scenario_id=scenario_id,
        frame_id=frame_id,
    )
    # 기준 위치 모드: 차량들의 평균 위치를 기준으로 재설정
    vehicles = payload.get("vehicles", [])
    if vehicles:
        xs = [float(v.get("position", {}).get("x", 0.0)) for v in vehicles]
        ys = [float(v.get("position", {}).get("y", 0.0)) for v in vehicles]
        zs = [float(v.get("position", {}).get("z", 0.0)) for v in vehicles]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cz = sum(zs) / len(zs)
        payload["reference_position"] = {"x": cx, "y": cy, "z": cz, "timestamp_s": None}
    if export_json_path:
        save_v2x_sim_payload(
            out_file=export_json_path,
            dataset_path=dataset_path,
            agent_id=agent_id,
            scenario_id=scenario_id,
            frame_id=frame_id,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    # 2) (보안) 공격 모드 설정: 검증 단계에서만 영향을 주도록 함
    # 주의: 공격 관측은 큐의 순서나 점수를 변경하지 않고, 오직 검증 단계에서만 영향을 줘야 함
    attack_mode_active = attack_mode
    # 3) JSON → 내부 표준 포맷으로 변환
    sample = build_sample_from_dict(payload)
    params = ScoringParams(
        max_range_m=250.0,
        congestion_weight=0.1,
    )
    selector = EgoSelector(
        scoring_params=params,
        min_validation_confidence=0.6,
        attack_mode=attack_mode_active,  # 공격 모드 전달
        no_defense=no_defense,  # No-defense 모드 전달
    )
    # center 모드에서는 거리≈0 제외 안전장치를 비활성화
    try:
        selector._exclude_zero_distance = False
    except Exception:
        pass
    
    # --- 검증용 detector_provider 주입 (selector.select() 호출 전에 필수) ---
    # ROBOSAC 설정이 있으면 검증 훅을 미리 준비
    _robosac_module = None
    if getattr(run_v2x_sim_example, "_robosac_cfg", None):
        robocfg = run_v2x_sim_example._robosac_cfg  # type: ignore[attr-defined]
        try:
            # 범용 함수를 사용하여 ROBOSAC 모듈 로드
            _robosac_module = _load_robosac_module(robocfg, PROJECT_ROOT)
        except Exception as e:
            print(f"[ROBOSAC] 모듈 로드 실패: {e}")
            _robosac_module = None
        
        # detector_provider 생성 및 주입
        if _robosac_module is not None:
            try:
                def _run_main_with_global(args_obj):
                    # baseline/egoloc 실행과 동일하게 모듈 레벨 args 설정
                    try:
                        _robosac_module.args = args_obj  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    return _robosac_module.main(args_obj)
                
                det_logs_root = (_P(PROJECT_ROOT) / "logs" / "detections").resolve()
                det_logs_root.mkdir(parents=True, exist_ok=True)
                
                def _read_exported(frame_indices: list[int], agent_id: int) -> dict:
                    """검증 결과 파일 읽기: agent ID 기반 파일명 사용 (agent{id}.json)"""
                    results = []
                    for fidx in frame_indices:
                        frame_dir = det_logs_root / f"scene_{scenario_id}" / f"frame_{fidx}"
                        fp = frame_dir / f"agent{agent_id}.json"  # agent ID 기반 파일명
                        if fp.exists():
                            try:
                                data = _json.loads(fp.read_text(encoding="utf-8"))
                                if isinstance(data, list):
                                    results.extend(data)
                            except Exception:
                                pass
                    return {
                        "collaborative_result": results if results else None,
                        "sent_representative_result": results if results else None,
                        "sent_per_peer": None,
                    }
                
                def _build_args_for_validation(ego_idx: int, frame_indices: list[int], attack_mode: str | None):
                    """검증용 ROBOSAC args 생성"""
                    args = _build_robosac_args_base(robocfg, scenario_id, attack_mode)
                    args.ego_agent = ego_idx
                    args.egoloc_attack_mode = attack_mode
                    args.egoloc_no_defense = no_defense
                    args.egoloc_selected_agent = None
                    # 검증용: 모든 후보가 frame 0, 1, 2, 3, 4를 사용하므로 항상 0부터 시작
                    args.sample_id = 0
                    args.max_frames_for_validation = 5  # frame 0, 1, 2, 3, 4 처리
                    # 검증용에서는 로그를 비활성화 (너무 많은 로그 생성 방지)
                    args.log = False
                    return args
                
                # 캐시: 같은 후보/프레임 조합에 대한 결과 재사용
                _detector_cache = {}
                
                def _detector_provider(*, candidate_vehicle_id: str, candidate_rank: int, frame_indices: list[int], attack_mode: str | None):
                    """
                    검증용 detector_provider: agent ID 기반으로 파일명 생성 (agent{id}.json)
                    candidate_vehicle_id: 검증할 차량 ID (예: "agent3")
                    candidate_rank: 우선순위 큐에서의 순위 (프레임 할당용, 1순위=0, 2순위=1, ...)
                    """
                    nonlocal _robosac_module  # 외부 스코프 변수 수정을 위해 반드시 필요
                    
                    # 실제 agent ID 파싱 (파일명 및 ROBOSAC 실행용)
                    ego_idx = _parse_agent_index(candidate_vehicle_id)
                    if ego_idx is None or _robosac_module is None:
                        return {}
                    
                    # 캐시 키: agent ID 기반으로 생성 (같은 agent는 같은 파일 사용)
                    cache_key = (ego_idx, tuple(frame_indices), attack_mode)
                    if cache_key in _detector_cache:
                        return _detector_cache[cache_key]
                    
                    scene_dir = det_logs_root / f"scene_{scenario_id}"
                    
                    # agent ID 기반 파일명으로 기존 결과 확인
                    missing_frames = []
                    for fidx in frame_indices:
                        frame_dir = scene_dir / f"frame_{fidx}"
                        fp = frame_dir / f"agent{ego_idx}.json"  # agent ID 기반 파일명
                        if not fp.exists():
                            missing_frames.append(fidx)
                    
                    # 모든 프레임 파일이 있으면 재사용
                    if not missing_frames:
                        exported = _read_exported(frame_indices, ego_idx)
                        if exported and exported.get("collaborative_result") is not None:
                            _detector_cache[cache_key] = exported
                            return exported
                    
                    # ROBOSAC 실행: agent ID 사용, export도 agent ID 기반 파일명으로
                    # 검증 모드 플래그 설정 (검증 단계에서만 JSON 생성하도록)
                    scene_dir = scene_dir.resolve()
                    _os.environ["EGOL0C_VALIDATION_MODE"] = "1"  # 검증 모드 활성화
                    _os.environ["EGOL0C_DET_EXPORT_DIR"] = str(scene_dir)
                    _os.environ["EGOL0C_CURRENT_EGO_IDX"] = str(ego_idx)  # agent ID
                    _os.environ["EGOL0C_DET_EXPORT_FRAMES"] = ",".join(map(str, frame_indices))
                    _os.environ["EGOL0C_ATTACK_MODE"] = str(attack_mode) if attack_mode else ""
                    
                    args = _build_args_for_validation(ego_idx, frame_indices, attack_mode)
                    try:
                        # 매 검증마다 완전한 상태 초기화를 수행하여 상태 누적 방지
                        _clear_cuda_cache()
                        
                        # 모듈을 다시 import하여 전역 상태 초기화
                        # 범용 함수를 사용하여 ROBOSAC 모듈 reload
                        if _robosac_module is not None:
                            try:
                                _robosac_module = _reload_robosac_modules_if_needed(_robosac_module, robocfg, PROJECT_ROOT)
                                # 모듈 내부 전역 변수 명시적 초기화
                                if hasattr(_robosac_module, 'args'):
                                    _robosac_module.args = None
                            except Exception as e:
                                print(f"[ROBOSAC] Module reload failed: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # baseline/egoloc 실행과 동일한 방식으로 실행
                        # _run_main_with_global이 내부에서 모듈 레벨 args를 설정함
                        if _robosac_module is not None:
                            _run_main_with_global(args)
                    except SystemExit:
                        pass
                    except Exception as e:
                        print(f"[detector_provider] ROBOSAC 실행 실패: {e}")
                        import traceback
                        traceback.print_exc()
                        return {}
                    finally:
                        # 검증 모드 플래그 해제 (다른 ROBOSAC 실행에 영향 없도록, 항상 정리)
                        _os.environ.pop("EGOL0C_VALIDATION_MODE", None)
                        _os.environ.pop("EGOL0C_CURRENT_EGO_IDX", None)
                        _os.environ.pop("EGOL0C_DET_EXPORT_DIR", None)
                        _os.environ.pop("EGOL0C_DET_EXPORT_FRAMES", None)
                        _os.environ.pop("EGOL0C_ATTACK_MODE", None)
                        
                        # 실행 후 다시 CUDA 캐시 정리
                        _clear_cuda_cache()
                    
                    # agent ID 기반 파일명으로 결과 읽기
                    exported = _read_exported(frame_indices, ego_idx)
                    if not exported or exported.get("collaborative_result") is None:
                        return {}
                    
                    _detector_cache[cache_key] = exported
                    return exported
                
                selector.detector_provider = _detector_provider
                print(f"[egoloc] 검증용 detector_provider 주입 완료")
            except Exception as e:
                print(f"[egoloc] detector_provider 주입 실패: {e}")
    
    # 검증 전 큐 출력: select()와 동일한 스코어링을 사용하기 위해 
    # neighbor_density와 communication_history를 미리 계산하여 전달
    # (이렇게 하면 _load_communication_history의 랜덤성으로 인한 순서 차이 방지)
    _neighbor_density = selector._compute_neighbor_density(sample)
    _communication_history = selector._load_communication_history()
    
    print("\n=== [검증 전] 우선순위 큐 ===")
    pre_validation_scores = selector._compute_queue_scores(sample, neighbor_density=_neighbor_density, communication_history=_communication_history)
    pre_queue_line = _format_candidate_queue_line(sample, scoring_params=params, vehicle_scores=pre_validation_scores)
    print(pre_queue_line)
    print()
    
    # EgoLoc 공격 시나리오가 활성화된 경우에만 검증 단계 이전에 후보 큐를 기반으로 공격자 리스트 생성
    # Baseline: agent0이 항상 공격자
    # EgoLoc: 큐의 상위 N개가 공격자가 됨 (1순위가 공격자)
    egoloc_attacker_list = []
    if attack_mode_active is not None:
        # 공격 시나리오가 활성화된 경우에만 공격자 리스트 생성
        if isinstance(pre_validation_scores, dict) and pre_validation_scores:
            # 점수 순으로 정렬하여 상위 N개를 공격자로 설정
            sorted_candidates = sorted(pre_validation_scores.items(), key=lambda x: x[1], reverse=True)
            # 공격자 수는 number_of_attackers로 설정 (기본값 1)
            num_attackers = robocfg.get("number_of_attackers", 1) if 'robocfg' in locals() else 1
            for i, (vehicle_id, score) in enumerate(sorted_candidates[:num_attackers]):
                agent_idx = _parse_agent_index(vehicle_id)
                if agent_idx is not None:
                    egoloc_attacker_list.append(agent_idx)
        
        # Baseline 실행을 위한 공격자 리스트도 설정 (ego_agent가 항상 공격자)
        # Baseline의 ego_agent는 robocfg에서 읽어오거나 기본값 1 사용
        baseline_ego_agent = robocfg.get("ego_agent", 1) if 'robocfg' in locals() else 1
        baseline_attacker_list = [baseline_ego_agent]  # Baseline은 항상 ego_agent가 공격자
        
        # 공격자 리스트를 환경 변수로 전달 (ROBOSAC에서 사용)
        # EgoLoc 검증 단계에서는 egoloc_attacker_list 사용, Baseline 실행 시에는 baseline_attacker_list 사용
        # 중요: Baseline 실행 시에는 EGOL0C_ATTACKER_LIST를 설정하지 않음 (EGOL0C_BASELINE_ATTACKER_LIST만 사용)
        if egoloc_attacker_list:
            _os.environ["EGOL0C_ATTACKER_LIST"] = ",".join(map(str, egoloc_attacker_list))
            print(f"[EgoLoc] 검증 전 공격자 리스트 생성 (공격 시나리오 활성화): {egoloc_attacker_list}")
        else:
            _os.environ.pop("EGOL0C_ATTACKER_LIST", None)
        
        # Baseline 실행을 위한 공격자 리스트도 별도 환경 변수로 설정
        # Baseline 실행 시에는 EGOL0C_BASELINE_ATTACKER_LIST만 사용하고, EGOL0C_ATTACKER_LIST는 제거
        # (egoloc_attack_handler에서 Baseline인지 EgoLoc인지 구분하기 위해)
        _os.environ["EGOL0C_BASELINE_ATTACKER_LIST"] = ",".join(map(str, baseline_attacker_list))
        
        # Baseline 실행 전에 EGOL0C_ATTACKER_LIST를 제거하여 Baseline이 EgoLoc로 인식되지 않도록 함
        # (Baseline 실행 시에는 EGOL0C_BASELINE_ATTACKER_LIST만 사용)
        # 주의: Baseline 실행 시에도 egoloc_attack_handler를 사용하지만, EGOL0C_ATTACKER_LIST가 없으면 Baseline으로 인식
        _os.environ.pop("EGOL0C_ATTACKER_LIST", None)
    else:
        # 공격 시나리오가 비활성화된 경우 환경 변수 제거
        _os.environ.pop("EGOL0C_ATTACKER_LIST", None)
        _os.environ.pop("EGOL0C_BASELINE_ATTACKER_LIST", None)
    
    # select() 호출 시 동일한 neighbor_density와 communication_history를 전달하여
    # 검증 전 큐와 실제 검증 순서가 일치하도록 보장
    result = selector.select(sample, neighbor_density=_neighbor_density, communication_history=_communication_history)
    print(json.dumps(_serialize_selection_result(result), ensure_ascii=False, indent=2))
    
    # 검증 후 큐 출력 (검증 결과 반영)
    print("\n=== [검증 후] 우선순위 큐 (최종 선택) ===")
    queue_line = _format_candidate_queue_line(sample, scoring_params=params, vehicle_scores=result.queue_order)
    print(queue_line)
    print(f"선택된 차량: {result.selected_vehicle_id}")
    print()

    # --- 큐 정보 로깅 (ROBOSAC 실행 전에 먼저 실행) ---
    if save_logs:
        logs_dir = (_P(PROJECT_ROOT) / "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 큐 정보를 별도 파일로 저장 (ROBOSAC 실행 여부와 관계없이)
        queue_log_path = logs_dir / f"egoloc_queue_{time_str()}.txt"
        # 공격자(큐 1위) 및 방어 성공/실패 판단
        try:
            top_id = None
            if isinstance(result.queue_order, dict) and result.queue_order:
                top_id = sorted(result.queue_order.items(), key=lambda x: x[1], reverse=True)[0][0]
            defense_failed = (top_id is not None and result.selected_vehicle_id == top_id)
        except Exception:
            top_id = None
            defense_failed = False
        
        queue_log_content = (
            f"=== EgoLoc Priority Queue ===\n"
            f"Scenario ID: {scenario_id}\n"
            f"Frame ID: {frame_id}\n"
            f"Attack Mode: {attack_mode_active}\n"
            f"Assumed Attacker (Top-1): {top_id}\n"
            f"Defense: {'FAIL' if defense_failed else 'SUCCESS'}\n"
            f"Selected Vehicle: {result.selected_vehicle_id}\n\n"
            f"[검증 전] Priority Queue:\n{pre_queue_line}\n\n"
            f"[검증 후] Priority Queue (최종 선택):\n{queue_line}\n"
        )
        queue_log_path.write_text(queue_log_content, encoding="utf-8")
        print(f"[egoloc] 큐 정보 로그 저장: {queue_log_path}")

    # --- 실제 ROBOSAC 기반 대역폭 측정 및 로깅 (선택사항) ---
    if log_bandwidth:
        try:
            # 실제 ROBOSAC 대역폭 로거 초기화: 절대 경로 사용
            bandwidth_path = (PROJECT_ROOT / "ROBOSAC" / "coperception").resolve()
            bandwidth_path_str = str(bandwidth_path)
            if bandwidth_path_str not in sys.path:
                sys.path.append(bandwidth_path_str)
            from coperception.utils.bandwidth_logger import init_bandwidth_logger
            
            bandwidth_logger = init_bandwidth_logger(
                log_dir="logs", 
                experiment_name=f"egoloc_{result.selected_vehicle_id}_real_bandwidth"
            )
            
            print(f"[bandwidth] EgoLoc 실제 ROBOSAC 대역폭 측정 초기화 완료")
            
            # CSV 파일 경로 설정
            sel_id = result.selected_vehicle_id
            out_dir = (_P(PROJECT_ROOT) / "logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            if sel_id and sel_id.startswith("agent"):
                try:
                    ego_idx = int(sel_id.replace("agent", ""))
                    ego_csv_name = f"egoloc_{ego_idx}_bandwidth.csv"
                except Exception:
                    ego_csv_name = f"egoloc_{sel_id}_bandwidth.csv"
            else:
                ego_csv_name = "egoloc_unknown_bandwidth.csv"

            ego_out_path = out_dir / ego_csv_name
            
        except Exception as e:
            print(f"[bandwidth] EgoLoc 대역폭 로거 초기화 실패: {e}")
            bandwidth_logger = None
            ego_out_path = None


    # ROBOSAC 연동: 선택된 자아 차량 기반 추가 실행 (옵션)
    if getattr(run_v2x_sim_example, "_robosac_cfg", None):
        robocfg = run_v2x_sim_example._robosac_cfg  # type: ignore[attr-defined]
        
        # 범용 함수를 사용하여 ROBOSAC 모듈 로드
        _robosac = _load_robosac_module(robocfg, PROJECT_ROOT)
        if _robosac is None:
            print("[robosac] ROBOSAC 모듈 로드 실패 - ROBOSAC 없이 EgoLoc 선택만 진행합니다")
        else:
            print(f"[robosac] ROBOSAC 모듈 로드 완료")

        # robosac 모듈 내부 함수가 전역 args에 의존하므로, main 호출 전 주입하는 래퍼
        def _run_main_with_global(args_obj):
            try:
                _robosac.args = args_obj  # type: ignore[attr-defined]
            except Exception:
                pass
            return _robosac.main(args_obj)

        # detector_provider는 이미 위에서 주입됨 (selector.select() 호출 전)

        # CUDA 디바이스 설정(요청 시)
        if robocfg.get("cuda_visible_devices") is not None:
            _os.environ["CUDA_VISIBLE_DEVICES"] = str(robocfg.get("cuda_visible_devices"))

        def _build_args(ego_idx: int | None) -> object:
            """baseline/egoloc 실행용 ROBOSAC args 생성"""
            args = _build_robosac_args_base(robocfg, scenario_id, attack_mode_active)
            args.ego_agent = ego_idx if ego_idx is not None else robocfg.get("ego_agent", 1)
            # attack_mode_active는 run_v2x_sim_example 함수의 로컬 변수이므로 직접 사용
            args.egoloc_attack_mode = attack_mode_active
            args.egoloc_no_defense = no_defense if 'no_defense' in locals() else False
            args.egoloc_selected_agent = result.selected_vehicle_id if 'result' in locals() else None
            # --log 플래그를 args.log에 반영하여 print_and_write_log가 로그 파일에 쓰도록 함
            args.log = save_logs if 'save_logs' in locals() else False
            # 디버깅: args 설정 확인
            print(f"[select_ego] _build_args: ego_idx={ego_idx}, attack_mode_active={attack_mode_active}, args.egoloc_attack_mode={args.egoloc_attack_mode}, args.log={args.log}")
            return args

        # 선택 결과에서 ego 인덱스 추출 (예: "agent3" → 3)
        sel_idx = _parse_agent_index(result.selected_vehicle_id)

        # Baseline 실제 ROBOSAC 대역폭 로깅
        if log_bandwidth:
            try:
                # Baseline 실제 ROBOSAC 대역폭 로거 초기화
                baseline_logger = init_bandwidth_logger(
                    log_dir="logs", 
                    experiment_name="baseline_real_bandwidth"
                )
                
                print(f"[bandwidth] Baseline 실제 ROBOSAC 대역폭 측정 초기화 완료")
                
                # Baseline CSV 파일 경로 설정
                baseline_out_path = out_dir / "baseline_bandwidth.csv"
                
            except Exception as e:
                print(f"[bandwidth] Baseline 대역폭 로거 초기화 실패: {e}")
                baseline_logger = None
                baseline_out_path = None

        print("[robosac] baseline 실행(기존 설정 ego)")
        baseline_text = None
        selected_text = None
        
        # ROBOSAC 실행을 안전하게 처리 (WSL 환경 대응)
        if _robosac is not None:
            try:
                # Baseline 대역폭 로거 설정 (ROBOSAC 실행 전)
                if log_bandwidth and 'baseline_logger' in locals() and baseline_logger is not None:
                    from coperception.utils.bandwidth_logger import set_bandwidth_logger
                    set_bandwidth_logger(baseline_logger)
                    print("[bandwidth] Baseline 대역폭 로거 설정 완료")
                
                # 실행 시간 측정 시작
                baseline_start_time = time.time()
                
                if save_logs:
                    _buf_base = _io.StringIO()
                    with _ctx.redirect_stdout(_buf_base):
                        try:
                            _run_main_with_global(_build_args(ego_idx=None))
                        except SystemExit:
                            pass
                        except Exception as e:
                            print(f"[robosac] baseline 실행 중 오류: {e}")
                    baseline_text = _buf_base.getvalue()
                else:
                    try:
                        _run_main_with_global(_build_args(ego_idx=None))
                    except Exception as e:
                        print(f"[robosac] baseline 실행 중 오류: {e}")
            except Exception as e:
                print(f"[robosac] baseline 실행 실패: {e}")
                print("[robosac] ROBOSAC 실행을 건너뜁니다.")
        else:
            print("[robosac] ROBOSAC 모듈이 없어 baseline 실행을 건너뜁니다.")
        
        # Baseline 실제 ROBOSAC 측정 결과 처리
        if log_bandwidth and 'baseline_logger' in locals() and baseline_logger is not None:
            try:
                # 실행 시간 측정 종료
                baseline_end_time = time.time()
                baseline_execution_time = baseline_end_time - baseline_start_time
                
                # Baseline 측정 데이터 확정
                baseline_logger.finalize_frame()
                baseline_summary = baseline_logger.get_summary()
                real_baseline_bpf = baseline_summary['total_bytes']
                baseline_frame_count = baseline_summary['frame_count']
                
                # 프레임 수 의미 명확화
                baseline_transmission_count = baseline_summary.get('transmission_count', baseline_frame_count)
                
                # 시간 기반 정규화 계산
                baseline_bytes_per_second = real_baseline_bpf / max(baseline_execution_time, 0.001)
                baseline_bytes_per_transmission = real_baseline_bpf / max(baseline_transmission_count, 1)
                baseline_bytes_per_actual_frame = real_baseline_bpf / 100  # ROBOSAC 고정 프레임 수
                
                print(f"[bandwidth] Baseline 실제 ROBOSAC 측정 결과: {real_baseline_bpf:,} bytes")
                print(f"[bandwidth] Baseline 실행 시간: {baseline_execution_time:.2f}초")
                print(f"[bandwidth] Baseline 전송 횟수: {baseline_transmission_count:,}")
                print(f"[bandwidth] Baseline 송신자 수: {baseline_summary['num_agents']}")
                print(f"[bandwidth] Baseline 초당 대역폭: {baseline_bytes_per_second:,.0f} bytes/sec")
                print(f"[bandwidth] Baseline 전송당 대역폭: {baseline_bytes_per_transmission:,.0f} bytes/transmission")
                print(f"[bandwidth] Baseline 실제 프레임당 대역폭: {baseline_bytes_per_actual_frame:,.0f} bytes/frame")
                
                # 실제 측정 결과로 CSV 파일 작성
                # Baseline: 고정된 agent0을 제외한 모든 차량이 송신자
                baseline_senders = len(sample.vehicles) - 1  # agent0 제외
                
                baseline_row = {
                    "timestamp": time_str(),
                    "selected_vehicle_id": "baseline",
                    "num_agents": len(sample.vehicles),
                    "senders": baseline_senders,  # 송신자 수 (5개)
                    "bytes_per_frame": int(real_baseline_bpf),
                    "frame_count": baseline_frame_count,
                    "bytes_per_frame_normalized": int(real_baseline_bpf / max(baseline_frame_count, 1)),
                    "execution_time_seconds": round(baseline_execution_time, 3),
                    "bytes_per_second": int(baseline_bytes_per_second),
                    "bytes_per_transmission": int(baseline_bytes_per_transmission),
                    "bytes_per_actual_frame": int(baseline_bytes_per_actual_frame),
                    "measurement_type": "real",
                }
                
                # CSV 파일에 기록
                write_header = not baseline_out_path.exists()
                with baseline_out_path.open("a", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=list(baseline_row.keys()))
                    if write_header:
                        w.writeheader()
                    w.writerow(baseline_row)
                    
            except Exception as e:
                print(f"[bandwidth] Baseline 실제 측정 결과 처리 실패: {e}")
        if sel_idx is not None:
            print(f"\n[robosac] selected ego 적용 실행(ego_agent={sel_idx})")
            # ROBOSAC 실행을 안전하게 처리 (WSL 환경 대응)
            if _robosac is not None:
                try:
                    # EgoLoc 실행 전에 환경 변수 재설정
                    # Baseline 실행 후 EGOL0C_ATTACKER_LIST가 제거되었으므로, EgoLoc 실행 시 다시 설정해야 함
                    if attack_mode_active is not None and 'egoloc_attacker_list' in locals() and egoloc_attacker_list:
                        _os.environ["EGOL0C_ATTACKER_LIST"] = ",".join(map(str, egoloc_attacker_list))
                        print(f"[EgoLoc] EgoLoc 실행 전 공격자 리스트 재설정: {egoloc_attacker_list}")
                        # Baseline 환경 변수는 제거하여 EgoLoc로 인식되도록 함
                        _os.environ.pop("EGOL0C_BASELINE_ATTACKER_LIST", None)
                    elif attack_mode_active is not None:
                        # 공격 시나리오가 활성화되었지만 egoloc_attacker_list가 없는 경우
                        _os.environ.pop("EGOL0C_ATTACKER_LIST", None)
                        _os.environ.pop("EGOL0C_BASELINE_ATTACKER_LIST", None)
                    
                    # EgoLoc 대역폭 로거 설정 (ROBOSAC 실행 전)
                    if log_bandwidth and 'bandwidth_logger' in locals() and bandwidth_logger is not None:
                        from coperception.utils.bandwidth_logger import set_bandwidth_logger
                        set_bandwidth_logger(bandwidth_logger)
                        print("[bandwidth] EgoLoc 대역폭 로거 설정 완료")
                    
                    # EgoLoc 실행 시간 측정 시작 (ROBOSAC 실행 직전)
                    egoloc_start_time = time.time()
                    
                    if save_logs:
                        _buf_sel = _io.StringIO()
                        with _ctx.redirect_stdout(_buf_sel):
                            try:
                                _run_main_with_global(_build_args(ego_idx=sel_idx))
                            except SystemExit:
                                pass
                            except Exception as e:
                                print(f"[robosac] selected ego 실행 중 오류: {e}")
                        selected_text = _buf_sel.getvalue()
                    else:
                        try:
                            _run_main_with_global(_build_args(ego_idx=sel_idx))
                        except Exception as e:
                            print(f"[robosac] selected ego 실행 중 오류: {e}")
                except Exception as e:
                    print(f"[robosac] selected ego 실행 실패: {e}")
                    print("[robosac] ROBOSAC 실행을 건너뜁니다.")
            else:
                print("[robosac] ROBOSAC 모듈이 없어 selected ego 실행을 건너뜁니다.")
            
            # EgoLoc 실제 ROBOSAC 측정 결과 처리
            if log_bandwidth and 'bandwidth_logger' in locals() and bandwidth_logger is not None:
                try:
                    # 실행 시간 측정 종료
                    egoloc_end_time = time.time()
                    egoloc_execution_time = egoloc_end_time - egoloc_start_time
                    
                    # EgoLoc 측정 데이터 확정
                    bandwidth_logger.finalize_frame()
                    bandwidth_summary = bandwidth_logger.get_summary()
                    real_bpf = bandwidth_summary['total_bytes']
                    egoloc_frame_count = bandwidth_summary['frame_count']
                    
                    # 프레임 수 의미 명확화
                    egoloc_transmission_count = bandwidth_summary.get('transmission_count', egoloc_frame_count)
                    
                    # 시간 기반 정규화 계산
                    egoloc_bytes_per_second = real_bpf / max(egoloc_execution_time, 0.001)
                    egoloc_bytes_per_transmission = real_bpf / max(egoloc_transmission_count, 1)
                    egoloc_bytes_per_actual_frame = real_bpf / 100  # ROBOSAC 고정 프레임 수
                    
                    print(f"[bandwidth] EgoLoc 실제 ROBOSAC 측정 결과: {real_bpf:,} bytes")
                    print(f"[bandwidth] EgoLoc 실행 시간: {egoloc_execution_time:.2f}초")
                    print(f"[bandwidth] EgoLoc 전송 횟수: {egoloc_transmission_count:,}")
                    print(f"[bandwidth] EgoLoc 송신자 수: {bandwidth_summary['num_agents']}")
                    print(f"[bandwidth] EgoLoc 초당 대역폭: {egoloc_bytes_per_second:,.0f} bytes/sec")
                    print(f"[bandwidth] EgoLoc 전송당 대역폭: {egoloc_bytes_per_transmission:,.0f} bytes/transmission")
                    print(f"[bandwidth] EgoLoc 실제 프레임당 대역폭: {egoloc_bytes_per_actual_frame:,.0f} bytes/frame")
                    
                    # 실제 측정 결과로 CSV 파일 작성
                    # EgoLoc: 선택된 자아 차량을 제외한 모든 차량이 송신자
                    egoloc_senders = len(sample.vehicles) - 1  # 선택된 자아 차량 제외
                    
                    row = {
                        "timestamp": time_str(),
                        "selected_vehicle_id": sel_id,
                        "num_agents": len(sample.vehicles),
                        "senders": egoloc_senders,  # 송신자 수 (5개)
                        "bytes_per_frame": int(real_bpf),
                        "frame_count": egoloc_frame_count,
                        "bytes_per_frame_normalized": int(real_bpf / max(egoloc_frame_count, 1)),
                        "execution_time_seconds": round(egoloc_execution_time, 3),
                        "bytes_per_second": int(egoloc_bytes_per_second),
                        "bytes_per_transmission": int(egoloc_bytes_per_transmission),
                        "bytes_per_actual_frame": int(egoloc_bytes_per_actual_frame),
                        "measurement_type": "real",
                    }
                    
                    # CSV 파일에 기록
                    write_header = not ego_out_path.exists()
                    with ego_out_path.open("a", newline="", encoding="utf-8") as f:
                        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
                        if write_header:
                            w.writeheader()
                        w.writerow(row)
                        
                except Exception as e:
                    print(f"[bandwidth] EgoLoc 실제 측정 결과 처리 실패: {e}")
            
            # 실제 ROBOSAC 기반 대역폭 절감 효과 분석 (다중 정규화 방식 적용)
            if log_bandwidth and 'baseline_logger' in locals() and 'bandwidth_logger' in locals():
                try:
                    baseline_summary = baseline_logger.get_summary()
                    egoloc_summary = bandwidth_logger.get_summary()
                    
                    baseline_bytes = baseline_summary['total_bytes']
                    egoloc_bytes = egoloc_summary['total_bytes']
                    
                    # 프레임 수 정규화를 위한 계산
                    baseline_frames = baseline_summary['frame_count']
                    egoloc_frames = egoloc_summary['frame_count']
                    
                    # 시간 기반 정규화 계산
                    baseline_time = baseline_execution_time
                    egoloc_time = egoloc_execution_time
                    
                    # 프레임당 대역폭 사용량 계산
                    baseline_bytes_per_frame = baseline_bytes / max(baseline_frames, 1)
                    egoloc_bytes_per_frame = egoloc_bytes / max(egoloc_frames, 1)
                    
                    # 시간당 대역폭 사용량 계산
                    baseline_bytes_per_second = baseline_bytes / max(baseline_time, 0.001)
                    egoloc_bytes_per_second = egoloc_bytes / max(egoloc_time, 0.001)
                    
                    # 공정한 비교를 위한 최소 프레임 수로 제한
                    min_frames = min(baseline_frames, egoloc_frames)
                    baseline_bytes_fair = baseline_bytes_per_frame * min_frames
                    egoloc_bytes_fair = egoloc_bytes_per_frame * min_frames
                    
                    if baseline_bytes_fair > 0:
                        reduction_percent = ((baseline_bytes_fair - egoloc_bytes_fair) / baseline_bytes_fair) * 100
                        reduction_bytes = baseline_bytes_fair - egoloc_bytes_fair
                        
                        print(f"\n[bandwidth] === 개선된 대역폭 절감 효과 분석 (다중 정규화) ===")
                        print(f"[bandwidth] 원본 데이터:")
                        print(f"[bandwidth]   Baseline: {baseline_bytes:,} bytes ({baseline_frames:,} 전송, {baseline_time:.2f}초)")
                        print(f"[bandwidth]   EgoLoc: {egoloc_bytes:,} bytes ({egoloc_frames:,} 전송, {egoloc_time:.2f}초)")
                        print(f"[bandwidth] 정규화된 대역폭 사용량:")
                        print(f"[bandwidth]   전송당: Baseline {baseline_bytes_per_frame:,.0f} vs EgoLoc {egoloc_bytes_per_frame:,.0f} bytes/transmission")
                        print(f"[bandwidth]   시간당: Baseline {baseline_bytes_per_second:,.0f} vs EgoLoc {egoloc_bytes_per_second:,.0f} bytes/sec")
                        print(f"[bandwidth] 공정한 비교 ({min_frames:,} 전송 기준):")
                        print(f"[bandwidth]   Baseline: {baseline_bytes_fair:,.0f} bytes")
                        print(f"[bandwidth]   EgoLoc: {egoloc_bytes_fair:,.0f} bytes")
                        print(f"[bandwidth]   절감량: {reduction_bytes:,.0f} bytes ({reduction_percent:.1f}%)")
                        
                        # 추가 분석: 효율성 지표
                        efficiency_ratio_transmission = egoloc_bytes_per_frame / max(baseline_bytes_per_frame, 1)
                        efficiency_ratio_time = egoloc_bytes_per_second / max(baseline_bytes_per_second, 1)
                        print(f"[bandwidth] 효율성 비율:")
                        print(f"[bandwidth]   전송당: {efficiency_ratio_transmission:.3f} (1.0보다 작을수록 효율적)")
                        print(f"[bandwidth]   시간당: {efficiency_ratio_time:.3f} (1.0보다 작을수록 효율적)")
                        
                        if reduction_percent > 0:
                            print(f"[bandwidth] ✅ EgoLoc이 Baseline 대비 {reduction_percent:.1f}% 대역폭 절감 달성!")
                        else:
                            print(f"[bandwidth] ⚠️ EgoLoc이 Baseline보다 더 많은 대역폭 사용")
                            
                except Exception as e:
                    print(f"[bandwidth] 대역폭 절감 효과 계산 실패: {e}")
                    
        else:
            print(f"[robosac] selected_vehicle_id를 ego 인덱스로 해석할 수 없어 생략: {sel_id}")

        # --log 지정 시 로그 파일 저장
        if save_logs:
            logs_dir = (_P(PROJECT_ROOT) / "logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            _suffix = "_mAP" if robocfg.get("mode") == "robosac_mAP" else ""
            
            # 큐 내용을 로그에 추가
            queue_line = _format_candidate_queue_line(sample, scoring_params=params, vehicle_scores=result.queue_order)
            queue_header = f"\n=== EgoLoc Priority Queue ===\n{queue_line}\n"
            
            base_path = logs_dir / f"baseline{_suffix}_{time_str()}.txt"
            if baseline_text is not None:
                # baseline 로그에 큐 정보 추가
                enhanced_baseline_text = baseline_text + queue_header
                base_path.write_text(enhanced_baseline_text, encoding="utf-8")
                print(f"[robosac] baseline 로그 저장: {base_path}")
            if sel_idx is not None and selected_text is not None:
                sel_path = logs_dir / f"egoloc_{sel_idx}{_suffix}_{time_str()}.txt"
                # egoloc 로그에 큐 정보 추가
                enhanced_selected_text = selected_text + queue_header
                sel_path.write_text(enhanced_selected_text, encoding="utf-8")
                print(f"[robosac] selected 로그 저장: {sel_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="자아 차량 선택 예제")
    parser.add_argument(
        "--mode", 
        choices=["sample", "v2x-sim"], 
        default="sample",
        help="실행 모드: sample (기본 JSON), v2x-sim (V2X-Sim 데이터셋)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="ROBOSAC/coperception/coperception/datasets/V2X-Sim-det/test",
        help="V2X-Sim 데이터셋 경로 (v2x-sim 모드에서 사용)"
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=0,
        help="에이전트 ID (v2x-sim 모드에서 사용)"
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=8,
        help="시나리오 ID (v2x-sim 모드에서 사용)"
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=0,
        help="프레임 ID (v2x-sim 모드에서 사용)"
    )
    parser.add_argument(
        "--reference-mode",
        choices=["center"],
        default="center",
        help="기준 위치 선택: center(차량 평균 위치)"
    )
    parser.add_argument(
        "--robosac",
        action="store_true",
        help="선택 결과를 기반으로 ROBOSAC을 추가 실행(기존 ego, 선택 ego 둘 다)"
    )
    parser.add_argument(
        "--robosac_mAP",
        action="store_true",
        help="ROBOSAC robosac_mAP 모드로 실행하여 mAP 측정 로그 저장(_mAP 접미사)"
    )
    parser.add_argument(
        "--robosac-data",
        type=str,
        default=str((PROJECT_ROOT / "ROBOSAC" / "coperception" / "coperception" / "datasets" / "V2X-Sim-det" / "test").resolve()),
        help="ROBOSAC 데이터 루트 경로"
    )
    parser.add_argument(
        "--robosac-resume",
        type=str,
        default=str((PROJECT_ROOT / "ROBOSAC" / "coperception" / "ckpt" / "meanfusion" / "epoch_49.pth")),
        help="ROBOSAC 체크포인트 경로"
    )
    parser.add_argument(
        "--robosac-com",
        type=str,
        default="mean",
        help="ROBOSAC 통신 모드(disco/when2com/v2v/sum/mean/max/cat/agent)"
    )
    parser.add_argument(
        "--robosac_k",
        type=int,
        default=3,
        help="ROBOSAC 합의 집합 크기 k (None이면 로직 오류가 발생할 수 있어 기본 3 권장)"
    )
    parser.add_argument(
        "--robosac_attackers",
        type=int,
        default=1,
        help="robosac_mAP 모드에서 공격자 수(desired_number_of_attackers)"
    )
    parser.add_argument(
        "--robosac_step_budget",
        type=int,
        default=3,
        help="robosac_mAP 모드에서 step_budget(desired_step_budget)"
    )
    parser.add_argument(
        "--robosac_adv_iter",
        type=int,
        default=15,
        help="robosac_mAP 모드에서 adversarial iterations"
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0",
        help="ROBOSAC 실행 시 CUDA_VISIBLE_DEVICES 값"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="ROBOSAC 실행 로그를 EgoLoc/logs/baseline.txt 및 egoloc_{idx}.txt로 저장"
    )
    parser.add_argument(
        "--attack",
        choices=["both", "first", "second", "subtle", "adaptive"],
        default=None,
        help="보안 공격 모드: both(두 공격), first(1위 부분 무시), second(2위 미묘한 차이), subtle(미묘한 조작), adaptive(적응적 공격)"
    )
    parser.add_argument(
        "--no-defense",
        action="store_true",
        help="No-defense 모드: 검증 단계를 건너뛰고 첫 번째 후보를 즉시 선택 (공격 효과 측정용)"
    )
    # 실제 대역폭 측정 플래그
    parser.add_argument("--log-bandwidth", action="store_true", help="실제 대역폭을 측정하여 logs/bandwidth.csv로 기록")
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="v2x-sim 모드에서 추출한 JSON을 파일로 저장(표준출력에도 표시)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        run_sample_example()
    elif args.mode == "v2x-sim":
        # ROBOSAC 설정 전달(있을 때만)
        if args.robosac or args.robosac_mAP:
            # 범용 함수를 사용하여 경로 정규화
            robosac_path = _normalize_robosac_path(None, PROJECT_ROOT)  # None이면 기본 경로 사용
            cfg = {
                "root_path": robosac_path,
                "data_path": args.robosac_data,
                "resume": args.robosac_resume,
                "com": args.robosac_com,
                "robosac_k": args.robosac_k,
                "mode": "robosac_mAP" if args.robosac_mAP else "robosac_validation",
                "number_of_attackers": args.robosac_attackers,
                "step_budget": args.robosac_step_budget,
                "adv_iter": args.robosac_adv_iter,
                "cuda_visible_devices": args.cuda_device,
                "ego_agent": 1,  # baseline의 자아 차량을 agent1으로 설정
                # 필요 시 추가: num_agent, scene_id 등
            }
            setattr(run_v2x_sim_example, "_robosac_cfg", cfg)
        run_v2x_sim_example(
            dataset_path=args.dataset_path,
            agent_id=args.agent_id,
            scenario_id=args.scenario_id,
            frame_id=args.frame_id,
            export_json_path=args.export_json,
            reference_mode=args.reference_mode,
            save_logs=args.log,
            attack_mode=args.attack,
        log_bandwidth=args.log_bandwidth,
            no_defense=args.no_defense,
        )


if __name__ == "__main__":
    main()