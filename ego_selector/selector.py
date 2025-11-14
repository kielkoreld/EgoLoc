from __future__ import annotations

from typing import Optional, List

from .types import (
    Candidate,
    DatasetSample,
    ScoringParams,
    SelectionResult,
)
from .scoring import compute_enhanced_comm_efficiency, _euclidean_distance
from .priority_queue import CandidatePriorityQueue


class EgoSelector:
    """
    위치 기반 스코어링 후 협력자 관측으로 검증하여 자아 차량을 선택하는 오케스트레이터.
    프레임워크 독립적으로 호출 가능.
    """

    def __init__(
        self,
        scoring_params: Optional[ScoringParams] = None,
        min_validation_confidence: float = 0.6,
        attack_mode: Optional[str] = None,
        no_defense: bool = False,
    ) -> None:
        self._params = scoring_params or ScoringParams()
        self._min_validation_confidence = min_validation_confidence
        self._attack_mode = attack_mode
        self._no_defense = no_defense  # 검증 단계를 건너뛰는 옵션
        # center 모드에서는 거리≈0 차량 제외 안전장치를 비활성화
        self._exclude_zero_distance: bool = False
        # 검증용 ROBOSAC 경량 실행 훅(옵션): Callable[[vehicle_id, candidate_rank, frame_indices, attack_mode], dict]
        # 외부에서 주입 가능하도록 공개 속성으로 둔다. 미주입 시 시뮬레이션 기반으로 동작한다.
        self.detector_provider = None  # type: ignore[assignment]
        # 프레임 샘플링: 모든 후보가 동일하게 frame 0, 1, 2, 3, 4를 사용
        self._validation_frames_per_candidate: int = 5
        self._validation_frame_count_cap: int = 100

    def _compute_queue_scores(self, sample: DatasetSample, neighbor_density: int | None = None, communication_history: dict | None = None) -> dict[str, float]:
        """스코어링만 수행하여 vehicle_id -> score 딕셔너리 반환 (검증 전 큐 확인용)
        
        Args:
            sample: 데이터셋 샘플
            neighbor_density: 이웃 밀도 (없으면 계산)
            communication_history: 통신 이력 (없으면 계산)
        """
        if neighbor_density is None:
            neighbor_density = self._compute_neighbor_density(sample)
        if communication_history is None:
            communication_history = self._load_communication_history()
        vehicle_scores = {}
        
        for i, vehicle in enumerate(sample.vehicles):
            base_score, meta = compute_enhanced_comm_efficiency(
                vehicle=vehicle,
                reference_position=sample.reference_position,
                params=self._params,
                scene_complexity=self._compute_scene_complexity(sample),
                vehicle_density=self._compute_vehicle_density(sample),
                neighbor_density=neighbor_density,
                communication_history=communication_history,
            )
            final_score = max(self._params.min_score_clip, min(self._params.max_score_clip, base_score))
            vehicle_scores[vehicle.vehicle_id] = final_score
        
        return vehicle_scores

    def select(self, sample: DatasetSample, neighbor_density: int | None = None, communication_history: dict | None = None) -> SelectionResult:
        """
        자아 차량 선택 수행
        
        Args:
            sample: 데이터셋 샘플
            neighbor_density: 이웃 밀도 (미제공 시 계산)
            communication_history: 통신 이력 (미제공 시 계산, 일관성을 위해 미리 계산하여 전달 권장)
        """
        queue = CandidatePriorityQueue()

        if neighbor_density is None:
            neighbor_density = self._compute_neighbor_density(sample)

        def _euclidean_distance_xyz(a, b) -> float:
            dx = a.x - b.x
            dy = a.y - b.y
            dz = a.z - b.z
            return (dx * dx + dy * dy + dz * dz) ** 0.5


        # 통신 이력을 한 번만 생성하여 일관성 보장 (미제공 시 계산)
        if communication_history is None:
            communication_history = self._load_communication_history()
        
        # 모든 차량의 점수를 미리 계산하여 저장
        vehicle_scores = {}
        candidates = []

        for i, vehicle in enumerate(sample.vehicles):
            # 향상된 스코어링 함수 사용
            base_score, meta = compute_enhanced_comm_efficiency(
                vehicle=vehicle,
                reference_position=sample.reference_position,
                params=self._params,
                scene_complexity=self._compute_scene_complexity(sample),
                vehicle_density=self._compute_vehicle_density(sample),
                neighbor_density=neighbor_density,
                communication_history=communication_history,
            )

            # 기본 점수 그대로 사용 (보정 제거)
            final_score = max(self._params.min_score_clip, min(self._params.max_score_clip, base_score))

            # 점수를 딕셔너리에 저장
            vehicle_scores[vehicle.vehicle_id] = final_score

            meta = dict(meta)
            meta.update({
                "base_score": base_score,
            })

            # 후보 객체 생성
            candidate = Candidate(vehicle=vehicle, score=final_score, metadata=meta)
            candidates.append(candidate)
        
        # 점수 순으로 정렬하여 큐에 삽입
        candidates.sort(key=lambda c: c.score, reverse=True)
        for candidate in candidates:
            queue.push(candidate)

        # 우선순위 큐 정보 생성: vehicle_id -> rank 매핑 (1순위=0, 2순위=1, ...)
        # detector_provider에서 사용할 수 있도록 저장
        queue_info = {}
        for rank, candidate in enumerate(candidates):
            queue_info[candidate.vehicle.vehicle_id] = rank
        
        # detector_provider에 우선순위 큐 정보 전달
        if hasattr(self, "detector_provider") and self.detector_provider:
            # detector_provider 함수 객체에 큐 정보 저장 (함수 객체는 속성 가질 수 있음)
            if hasattr(self.detector_provider, "__self__") or hasattr(self.detector_provider, "queue_info"):
                try:
                    self.detector_provider.queue_info = queue_info  # type: ignore[attr-defined]
                except Exception:
                    pass

        audited = 0
        best_candidate: Optional[Candidate] = None
        best_validation = None

        # 신뢰도(reliability)를 임계값과 비교하여 검증 수행
        min_conf = 0.6
        try:
            # 하위호환: 생성자 인자의 의미를 유지하기 위해 내부 속성명을 사용하지 않음
            # 외부에서 임계값을 조정할 경우를 대비해 동적 조회(없으면 기본 0.6)
            min_conf = float(getattr(self, "_min_validation_confidence", 0.6))
        except Exception:
            min_conf = 0.6

        while len(queue) > 0:
            cand = queue.pop()
            if best_candidate is None:
                best_candidate = cand
            audited += 1
            candidate_rank = audited - 1  # 1순위=0, 2순위=1, ...
            
            # 디버깅: 검증 순서 확인
            print(f"[검증 순서] {audited}순위 후보 검증 시작: {cand.vehicle.vehicle_id} (점수: {cand.score:.2f})")

            # No-defense 모드: 검증 단계를 건너뛰고 첫 번째 후보를 즉시 선택
            if self._no_defense:
                from .types import ValidationResult
                validation = ValidationResult(
                    target_vehicle_id=cand.vehicle.vehicle_id,
                    is_valid=True,  # 강제로 통과
                    confidence=1.0,
                    reasons=["no_defense_mode_skipped_validation"],
                )
                print(f"[No-Defense] 검증 단계 건너뛰고 {cand.vehicle.vehicle_id} 선택 (순위: {audited})")
                return SelectionResult(
                    selected_vehicle_id=cand.vehicle.vehicle_id,
                    candidate=cand,
                    validation=validation,
                    audited_count=audited,
                    queue_order=vehicle_scores,
                )
            
            # 보안 모듈 검증 (포함 증명 + 협력자 합의)
            security_results = self._perform_security_validation(cand.vehicle.vehicle_id, sample, candidate_rank)
            
            # 기본 신뢰도 검증
            # 신뢰도 계산 제거됨
            
            # 최종 검증 결과 결정 (신뢰도 검증 제거)
            is_valid = security_results["overall_valid"]
            reasons = []
            
            if not security_results["overall_valid"]:
                reasons.extend(security_results["failure_reasons"])
            # 신뢰도 검증 제거됨
            
            from .types import ValidationResult  # 지역 임포트로 순환 의존 회피
            validation = ValidationResult(
                target_vehicle_id=cand.vehicle.vehicle_id,
                is_valid=is_valid,
                confidence=1.0,  # 신뢰도 계산 제거로 기본값 사용
                reasons=reasons,
            )

            if is_valid:
                return SelectionResult(
                    selected_vehicle_id=cand.vehicle.vehicle_id,
                    candidate=cand,
                    validation=validation,
                    audited_count=audited,
                    queue_order=vehicle_scores,
                )
            if best_validation is None or validation.confidence > best_validation.confidence:
                best_validation = validation

        return SelectionResult(
            selected_vehicle_id=best_candidate.vehicle.vehicle_id if best_candidate else None,
            candidate=best_candidate,
            validation=best_validation,
            audited_count=audited,
            queue_order=vehicle_scores,
        )
    
    def _compute_scene_complexity(self, sample: DatasetSample) -> float:
        """장면 복잡도 계산"""
        if len(sample.vehicles) <= 1:
            return 0.0
        
        # 차량 간 거리 분산을 기반으로 복잡도 계산
        distances = []
        for i, v1 in enumerate(sample.vehicles):
            for j, v2 in enumerate(sample.vehicles):
                if i != j:
                    dist = _euclidean_distance(v1.position, v2.position)
                    distances.append(dist)
        
        if not distances:
            return 0.0
        
        mean_dist = sum(distances) / len(distances)
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        
        # 정규화된 복잡도 (0-1)
        complexity = min(1.0, variance / (mean_dist ** 2 + 1e-6))
        return complexity
    
    def _compute_vehicle_density(self, sample: DatasetSample) -> float:
        """차량 밀도 계산"""
        if len(sample.vehicles) <= 1:
            return 0.0
        
        # 차량들이 차지하는 영역 계산
        positions = [v.position for v in sample.vehicles]
        min_x = min(p.x for p in positions)
        max_x = max(p.x for p in positions)
        min_y = min(p.y for p in positions)
        max_y = max(p.y for p in positions)
        
        area = (max_x - min_x) * (max_y - min_y)
        if area <= 0:
            return 1.0
        
        # 차량 수 대비 영역 밀도
        density = len(sample.vehicles) / max(area, 1.0)
        return min(1.0, density)
    
    def _load_communication_history(self) -> dict:
        """통신 이력 로드 - 시뮬레이션된 데이터 생성"""
        import time
        import random
        
        # 현재 시간 기준으로 시뮬레이션된 통신 이력 생성
        current_time = time.time()
        history = {}
        
        # 각 차량별로 통신 이력 시뮬레이션
        for i in range(6):  # agent0 ~ agent5
            vehicle_id = f"agent{i}"
            records = []
            
            # 최근 30분간의 통신 기록 생성 (5분 간격)
            for j in range(6):
                timestamp = current_time - (j * 5 * 60)  # 5분 간격
                # 차량별로 다른 통신 패턴 시뮬레이션
                base_bytes = 50 * 1024 * 1024  # 50MB 기본
                variation = random.uniform(0.5, 1.5)  # ±50% 변동
                bytes_transmitted = int(base_bytes * variation)
                
                # agent4는 높은 통신량, agent5는 낮은 통신량으로 설정
                if vehicle_id == "agent4":
                    bytes_transmitted = int(bytes_transmitted * 1.5)  # 50% 더 많은 통신
                elif vehicle_id == "agent5":
                    bytes_transmitted = int(bytes_transmitted * 0.7)  # 30% 적은 통신
                
                records.append({
                    'timestamp': timestamp,
                    'bytes_transmitted': bytes_transmitted,
                    'success': random.random() > 0.1,  # 90% 성공률
                    'latency_ms': random.uniform(10, 100)
                })
            
            history[vehicle_id] = records
        
        return history
    
    def _compute_neighbor_density(self, sample: DatasetSample) -> int:
        """
        통신 범위 내 실제 이웃 차량 수 계산
        
        Args:
            sample: 데이터셋 샘플
            
        Returns:
            통신 범위 내 이웃 차량 수
        """
        if len(sample.vehicles) <= 1:
            return 0
        
        # 통신 범위 설정 (파라미터에서 가져오기)
        max_range_m = self._params.max_range_m
        
        # 각 차량별로 통신 범위 내 이웃 수 계산
        neighbor_counts = []
        
        for i, vehicle in enumerate(sample.vehicles):
            neighbor_count = 0
            
            # 다른 모든 차량과의 거리 계산
            for j, other_vehicle in enumerate(sample.vehicles):
                if i != j:  # 자기 자신 제외
                    distance = _euclidean_distance(vehicle.position, other_vehicle.position)
                    if distance <= max_range_m:
                        neighbor_count += 1
            
            neighbor_counts.append(neighbor_count)
        
        # 전체 차량의 평균 이웃 수 반환
        avg_neighbor_count = sum(neighbor_counts) / len(neighbor_counts) if neighbor_counts else 0
        return int(round(avg_neighbor_count))
    
    def _perform_security_validation(self, vehicle_id: str, sample: DatasetSample, candidate_rank: int) -> dict:
        """
        보안 모듈 검증 수행 (포함 증명 + 협력자 합의)
        
        Args:
            vehicle_id: 검증할 차량 ID
            sample: 데이터셋 샘플
            candidate_rank: 후보 순위 (0부터 시작)
            
        Returns:
            검증 결과 딕셔너리
        """
        from .validator import prove_inclusion, peer_consensus_certification
        
        # 1. 포함 증명 검증
        inclusion_result = self._validate_inclusion_proof(vehicle_id, sample, candidate_rank)
        
        # 2. 협력자 합의 검증
        consensus_result = self._validate_peer_consensus(vehicle_id, sample, candidate_rank)
        
        # 3. 공격 시나리오 적용
        if self._attack_mode:
            inclusion_result, consensus_result = self._apply_attack_scenario(
                vehicle_id, sample, candidate_rank, inclusion_result, consensus_result
            )
        
        # 4. 결과 출력
        self._print_security_results(vehicle_id, candidate_rank, inclusion_result, consensus_result)
        
        # 5. 최종 결과 결정
        overall_valid = inclusion_result["is_valid"] and consensus_result["is_valid"]
        failure_reasons = []
        
        if not inclusion_result["is_valid"]:
            failure_reasons.append("inclusion_proof_failed")
        if not consensus_result["is_valid"]:
            failure_reasons.append("peer_consensus_failed")
        
        return {
            "overall_valid": overall_valid,
            "inclusion_proof": inclusion_result,
            "peer_consensus": consensus_result,
            "failure_reasons": failure_reasons
        }
    
    def _validate_inclusion_proof(self, vehicle_id: str, sample: DatasetSample, candidate_rank: int = None) -> dict:
        """포함 증명 검증: 협력 차량의 정보가 후보 차량의 협업 인식에 제대로 반영되는지 확인"""
        from .validator import prove_inclusion
        
        # 실제 감지자 훅이 제공되면 우선 사용 (ROBOSAC 경량 실행 등)
        # 같은 후보에 대한 검증 결과는 캐싱하여 재사용
        collaborative_result = None
        sent_collaborative_result = None
        
        if getattr(self, "detector_provider", None) and candidate_rank is not None:
            try:
                # 프레임 샘플링: 모든 후보가 동일하게 frame 0, 1, 2, 3, 4를 사용
                start = 0
                frames = list(range(start, min(start + self._validation_frames_per_candidate, self._validation_frame_count_cap)))
                
                det_provider_result = self.detector_provider(  # type: ignore[misc]
                    candidate_vehicle_id=vehicle_id,
                    candidate_rank=candidate_rank,
                    frame_indices=frames,
                    attack_mode=self._attack_mode,
                )
                
                if isinstance(det_provider_result, dict):
                    collaborative_result = det_provider_result.get("collaborative_result")
                    sent_collaborative_result = det_provider_result.get("sent_representative_result")
            except Exception as e:
                print(f"[egoloc] detector_provider 호출 실패: vehicle_id={vehicle_id}, rank={candidate_rank}, error={e}")
                det_provider_result = None
        
        # 1) 훅이 없거나 실패 시 기존 시뮬레이션 융합 결과 사용
        if collaborative_result is None:
            collaborative_result = self._generate_fused_detections(sample)
        
        # 후보 차량이 협력자들에게 전송한 협업 인식 결과들 중 하나를 대표로 선택
        # (실제로는 모든 협력자에게 동일한 결과를 전송해야 함)
        representative_peer_id = None
        for v in sample.vehicles:
            if v.vehicle_id != vehicle_id:
                representative_peer_id = v.vehicle_id
                break
        
        if representative_peer_id is None:
            # 협력자가 없는 경우
            return {
                "is_valid": False,
                "match_ratio": 0.0,
                "total_contributed": 0,
                "total_matched": 0,
                "details": {"reason": "no_peers_available"}
            }
        
        # 후보 차량이 협력자에게 전송한 협업 인식 결과
        # detector_provider 결과가 없으면 시뮬레이션 결과 사용
        if sent_collaborative_result is None:
            sent_collaborative_result = self._generate_collaborative_result_for_peer(
                candidate_vehicle_id=vehicle_id,
                peer_vehicle_id=representative_peer_id,
                sample=sample,
                candidate_rank=candidate_rank
            )
        
        # 공격 시나리오 확인: 악성 에이전트가 협력자 정보를 무시하는 경우 (큐의 실제 순서 기반)
        is_malicious_ignoring_peers = (candidate_rank == 0 and self._attack_mode in ("first", "both"))
        is_malicious_subtle = (candidate_rank == 0 and self._attack_mode == "subtle")
        
        if is_malicious_ignoring_peers:
            # 악성 에이전트: 협력자들의 정보를 부분적으로 무시하고 조작된 결과 생성
            sent_collaborative_result = self._generate_malicious_random_results(vehicle_id, sample)
        elif is_malicious_subtle:
            # 미묘한 조작 공격: 정상 결과와 유사하지만 약간의 조작
            sent_collaborative_result = self._generate_malicious_subtle_results(vehicle_id, sample)
        
        # 포함 증명 검증: 협력자들의 기여 정보가 협업 인식 결과에 반영되었는지 확인
        proof_result = prove_inclusion(
            contributed_detections=collaborative_result,  # 협력자들의 기여 정보 (융합된 결과)
            fused_detections=sent_collaborative_result,  # 후보 차량이 전송한 협업 인식 결과
            iou_threshold=0.7,  # 더 엄격한 IoU 임계값
            min_match_ratio=0.95  # 95% 이상 매칭 필요 (매우 엄격하게)
        )
        
        return {
            "is_valid": proof_result.is_included,
            "match_ratio": proof_result.match_ratio,
            "total_contributed": proof_result.total_contributed,
            "total_matched": proof_result.total_matched,
            "details": proof_result.details
        }
    
    def _validate_peer_consensus(self, vehicle_id: str, sample: DatasetSample, candidate_rank: int = None) -> dict:
        """협력자 합의 검증: 협력 차량들이 후보 차량으로부터 동일한 협업 인식 결과를 받았는지 확인"""
        from .validator import peer_consensus_certification
        
        # 후보 차량이 각 협력자에게 전송한 협업 인식 결과 시뮬레이션
        peers_received_results = {}
        for v in sample.vehicles:
            if v.vehicle_id == vehicle_id:
                continue  # 후보 차량 자신은 제외
            # 후보 차량이 해당 협력자에게 전송한 협업 인식 결과
            peers_received_results[v.vehicle_id] = self._generate_collaborative_result_for_peer(
                candidate_vehicle_id=vehicle_id, 
                peer_vehicle_id=v.vehicle_id, 
                sample=sample,
                candidate_rank=candidate_rank
            )
        # 실제 감지자 훅이 제공되면, 각 피어에 대한 전송 결과를 덮어씀
        # 주의: detector_provider는 _validate_inclusion_proof에서 이미 호출되었으므로,
        # 여기서는 같은 프레임 세트를 재사용하되, 이미 캐싱된 결과가 있으면 사용
        # (detector_provider 내부 캐싱에 의존)
        # 현재 구현에서는 sent_per_peer가 제공되지 않으므로, 시뮬레이션 결과 사용
        
        # 협력자 합의 검증: 협력자들이 받은 결과가 서로 일치하는지 확인
        consensus_result = peer_consensus_certification(
            peers_results=peers_received_results,
            exclude_agent_id=vehicle_id,  # 후보 차량은 제외
            iou_threshold=0.7,  # 더 엄격한 IoU 임계값
            max_unmatched_ratio=0.05,  # 5%까지 불일치 허용 (매우 엄격하게)
            quorum_ratio=0.95  # 95% 이상 합의 필요 (매우 엄격)
        )
        
        return {
            "is_valid": consensus_result.consensus,
            "num_certified": consensus_result.num_certified,
            "total_peers": consensus_result.total_peers,
            "per_peer_unmatched": consensus_result.per_peer_unmatched,
            "threshold": consensus_result.threshold
        }
    
    def _apply_attack_scenario(self, vehicle_id: str, sample: DatasetSample, candidate_rank: int, inclusion_result: dict, consensus_result: dict) -> tuple:
        """공격 시나리오 적용 - 악성 에이전트의 협업 인식 결과를 조작 (큐의 실제 순서 기반)"""
        if candidate_rank == 0:  # 첫 번째 후보 (악성 에이전트)
            if self._attack_mode == "first":
                # 악성 에이전트: 협력자들의 정보를 부분적으로 무시하고 조작된 협업 인식 결과 생성 (포함 증명 실패)
                inclusion_result = {
                    "is_valid": False,
                    "match_ratio": 0.2,  # 일부만 매칭 (더 현실적)
                    "total_contributed": 5,
                    "total_matched": 1,
                    "details": {"reason": "malicious_partial_ignores_peer_contributions"}
                }
            elif self._attack_mode == "second":
                # 악성 에이전트: 협력자별로 미묘하게 다른 협업 인식 결과 전송 (협력자 합의 실패)
                consensus_result = {
                    "is_valid": False,
                    "num_certified": 2,  # 일부만 인증 (더 현실적)
                    "total_peers": 5,
                    "per_peer_unmatched": {"agent1": 0.3, "agent2": 0.4, "agent4": 0.2},
                    "threshold": 0.3
                }
            elif self._attack_mode == "both":
                # 악성 에이전트: 협력자 정보 무시 + 협력자별 다른 결과 전송
                # 실제 공격 함수를 호출하여 동적으로 결과 생성
                inclusion_result = self._validate_inclusion_proof(vehicle_id, sample, candidate_rank)
                consensus_result = self._validate_peer_consensus(vehicle_id, sample, candidate_rank)
            elif self._attack_mode == "subtle":
                # 미묘한 조작 공격: 정상 결과와 유사하지만 약간의 조작
                # 실제 공격 함수를 호출하여 동적으로 결과 생성
                inclusion_result = self._validate_inclusion_proof(vehicle_id, sample, candidate_rank)
                consensus_result = self._validate_peer_consensus(vehicle_id, sample, candidate_rank)
            elif self._attack_mode == "adaptive":
                # 적응적 공격: 보안 검증을 우회하려는 시도
                # 실제 공격 함수를 호출하여 동적으로 결과 생성
                inclusion_result = self._validate_inclusion_proof(vehicle_id, sample, candidate_rank)
                consensus_result = self._validate_peer_consensus(vehicle_id, sample, candidate_rank)
        
        elif candidate_rank == 1 and self._attack_mode == "both":
            # 두 번째 후보도 악성 에이전트로 설정 (both 모드에서만)
            consensus_result = {
                "is_valid": False,
                "num_certified": 1,
                "total_peers": 5,
                "per_peer_unmatched": {"agent0": 0.3, "agent2": 0.4, "agent4": 0.2},
                "threshold": 0.3
            }
        
        return inclusion_result, consensus_result
    
    def _print_security_results(self, vehicle_id: str, candidate_rank: int, inclusion_result: dict, consensus_result: dict):
        """보안 검증 결과 출력"""
        rank_name = ["1st", "2nd", "3rd", "4th", "5th", "6th"][candidate_rank] if candidate_rank < 6 else f"{candidate_rank + 1}th"
        
        print(f"[Security Validation] {rank_name} candidate {vehicle_id}:")
        
        # 포함 증명 결과
        inclusion_status = "PASS" if inclusion_result["is_valid"] else "FAIL"
        print(f"  Inclusion Proof: {inclusion_status}")
        print(f"    - 협력자 정보가 협업 인식에 반영되었는지 확인")
        print(f"    - Match ratio: {inclusion_result['match_ratio']:.2f}")
        print(f"    - Contributed: {inclusion_result['total_contributed']}, Matched: {inclusion_result['total_matched']}")
        
        # 협력자 합의 결과
        consensus_status = "PASS" if consensus_result["is_valid"] else "FAIL"
        print(f"  Peer Consensus: {consensus_status}")
        print(f"    - 협력자들이 동일한 협업 인식 결과를 받았는지 확인")
        print(f"    - Certified: {consensus_result['num_certified']}/{consensus_result['total_peers']}")
        
        print()
    
    def _generate_realistic_detections(self, vehicle_id: str, sample: DatasetSample) -> list[dict]:
        """고정된 검출 결과 생성 (일관된 보안 검증을 위해)"""
        # 해당 차량의 위치를 기준으로 검출 결과 생성
        vehicle = next((v for v in sample.vehicles if v.vehicle_id == vehicle_id), None)
        if not vehicle:
            return []
        
        # 공통 관심 영역 계산 (모든 차량의 중심점)
        center_x = sum(v.position.x for v in sample.vehicles) / len(sample.vehicles)
        center_y = sum(v.position.y for v in sample.vehicles) / len(sample.vehicles)
        
        # 고정된 객체들 생성 (차량 ID 기반으로 일관된 결과)
        vehicle_hash = hash(vehicle_id) % 1000
        num_detections = 3 + (vehicle_hash % 3)  # 3-5개
        
        detections = []
        for i in range(num_detections):
            # 차량별 고정된 오프셋 (차량 ID 기반)
            offset_x = (vehicle_hash + i * 17) % 40 - 20  # -20 ~ 20
            offset_y = (vehicle_hash + i * 23) % 30 - 15  # -15 ~ 15
            
            x = center_x + offset_x
            y = center_y + offset_y
            width = 25 + (vehicle_hash + i) % 10  # 25-35
            height = 18 + (vehicle_hash + i * 7) % 8  # 18-26
            
            # 고정된 신뢰도
            confidence = 0.8 + (vehicle_hash + i) % 20 * 0.01  # 0.8-0.99
            
            detection = {
                "x1": x,
                "y1": y,
                "x2": x + width,
                "y2": y + height,
                "confidence": confidence,
                "class": "car",
                "source": vehicle_id
            }
            detections.append(detection)
        
        return detections
    
    def _generate_fused_detections(self, sample: DatasetSample) -> list[dict]:
        """고정된 협업 인식 결과 생성 (일관된 보안 검증을 위해)"""
        # 모든 차량의 검출 결과를 수집
        all_detections = []
        for vehicle in sample.vehicles:
            vehicle_detections = self._generate_realistic_detections(vehicle.vehicle_id, sample)
            all_detections.extend(vehicle_detections)
        
        # 협업 인식 결과 생성 (중복 제거 및 융합)
        fused_detections = self._fuse_detections(all_detections)
        
        return fused_detections
    
    def _fuse_detections(self, detections: list[dict]) -> list[dict]:
        """고정된 검출 결과 융합 (일관된 보안 검증을 위해)"""
        if not detections:
            return []
        
        # IoU 기반 클러스터링으로 중복 제거
        clusters = self._cluster_detections(detections, iou_threshold=0.5)
        
        fused_detections = []
        for cluster in clusters:
            if len(cluster) == 1:
                # 단일 검출 결과 (노이즈 없이 그대로 사용)
                detection = cluster[0].copy()
                detection["source"] = "fused"
                fused_detections.append(detection)
            else:
                # 여러 검출 결과의 평균
                avg_x1 = sum(d["x1"] for d in cluster) / len(cluster)
                avg_y1 = sum(d["y1"] for d in cluster) / len(cluster)
                avg_x2 = sum(d["x2"] for d in cluster) / len(cluster)
                avg_y2 = sum(d["y2"] for d in cluster) / len(cluster)
                avg_confidence = sum(d["confidence"] for d in cluster) / len(cluster)
                
                # 협업으로 인한 신뢰도 향상
                improved_confidence = min(0.95, avg_confidence + 0.1)
                
                fused_detection = {
                    "x1": avg_x1,
                    "y1": avg_y1,
                    "x2": avg_x2,
                    "y2": avg_y2,
                    "confidence": improved_confidence,
                    "class": "car",
                    "source": "fused"
                }
                fused_detections.append(fused_detection)
        
        return fused_detections
    
    def _cluster_detections(self, detections: list[dict], iou_threshold: float = 0.5) -> list[list[dict]]:
        """IoU 기반 검출 결과 클러스터링"""
        from .validator import _iou, _to_xyxy
        
        if not detections:
            return []
        
        # 박스를 (x1,y1,x2,y2) 형태로 변환
        boxes_with_data = []
        for detection in detections:
            xyxy = _to_xyxy(detection)
            if xyxy is not None:
                boxes_with_data.append((xyxy, detection))
        
        clusters = []
        used = [False] * len(boxes_with_data)
        
        for i, (box1, detection1) in enumerate(boxes_with_data):
            if used[i]:
                continue
            
            cluster = [detection1]
            used[i] = True
            
            # IoU가 임계값 이상인 다른 박스들을 찾아서 클러스터에 추가
            for j, (box2, detection2) in enumerate(boxes_with_data):
                if used[j]:
                    continue
                
                iou_score = _iou(box1, box2)
                if iou_score >= iou_threshold:
                    cluster.append(detection2)
                    used[j] = True
            
            clusters.append(cluster)
        
        return clusters
    
    def _generate_collaborative_result_for_peer(self, candidate_vehicle_id: str, peer_vehicle_id: str, sample: DatasetSample, candidate_rank: int = None) -> list[dict]:
        """후보 차량이 특정 협력자에게 전송할 협업 인식 결과 생성"""
        # 후보 차량의 위치를 기준으로 협업 인식 결과 생성
        candidate_vehicle = next((v for v in sample.vehicles if v.vehicle_id == candidate_vehicle_id), None)
        if not candidate_vehicle:
            return []
        
        # 협력자 차량의 위치를 고려한 협업 인식 결과 생성
        peer_vehicle = next((v for v in sample.vehicles if v.vehicle_id == peer_vehicle_id), None)
        if not peer_vehicle:
            return []
        
        # 후보 차량과 협력자 차량 간의 거리 계산
        from .scoring import _euclidean_distance
        distance = _euclidean_distance(candidate_vehicle.position, peer_vehicle.position)
        
        # 거리에 따른 협업 인식 결과 품질 조정
        if distance > 200.0:  # 200m 이상이면 품질 저하
            quality_factor = 0.7
        elif distance > 100.0:  # 100-200m면 약간 품질 저하
            quality_factor = 0.85
        else:  # 100m 이하면 좋은 품질
            quality_factor = 1.0
        
        # 후보 차량이 생성한 협업 인식 결과 (모든 협력자의 정보를 융합한 결과)
        collaborative_result = self._generate_fused_detections(sample)
        
        # 공격 시나리오 확인 (큐의 실제 순서 기반)
        if candidate_rank is None:
            candidate_rank = self._get_candidate_rank(candidate_vehicle_id, sample)
        
        is_malicious_different = (candidate_rank == 0 and self._attack_mode in ("second", "both"))
        is_malicious_adaptive = (candidate_rank == 0 and self._attack_mode == "adaptive")
        
        if is_malicious_different:
            # 악성 에이전트: 협력자별로 미묘하게 다른 협업 인식 결과 생성
            return self._generate_malicious_different_results(candidate_vehicle_id, peer_vehicle_id, sample, collaborative_result)
        elif is_malicious_adaptive:
            # 적응적 공격: 보안 검증을 우회하려는 시도
            return self._generate_malicious_adaptive_results(candidate_vehicle_id, peer_vehicle_id, sample, collaborative_result)
        else:
            # 정상 에이전트: 협력자별로 약간씩 다른 결과를 생성 (실제 통신에서는 노이즈나 지연으로 인해 미세한 차이가 발생)
            import random
            random.seed(hash(candidate_vehicle_id + peer_vehicle_id) % 1000)  # 일관된 결과를 위해 시드 설정
            
            adjusted_result = []
            for detection in collaborative_result:
                # 품질에 따른 신뢰도 조정
                adjusted_confidence = detection["confidence"] * quality_factor
                
                # 미세한 위치 노이즈 추가 (실제 통신 환경 시뮬레이션)
                noise_x = random.uniform(-0.5, 0.5)
                noise_y = random.uniform(-0.5, 0.5)
                
                adjusted_detection = {
                    "x1": detection["x1"] + noise_x,
                    "y1": detection["y1"] + noise_y,
                    "x2": detection["x2"] + noise_x,
                    "y2": detection["y2"] + noise_y,
                    "confidence": adjusted_confidence,
                    "class": detection["class"],
                    "source": f"collaborative_from_{candidate_vehicle_id}_to_{peer_vehicle_id}"
                }
                adjusted_result.append(adjusted_detection)
            
            return adjusted_result
    
    def _get_candidate_rank(self, vehicle_id: str, sample: DatasetSample) -> int:
        """후보 차량의 순위를 반환 (0부터 시작)"""
        # 점수 기반으로 순위 계산
        vehicle_scores = {}
        for v in sample.vehicles:
            base_score, _ = compute_enhanced_comm_efficiency(
                vehicle=v,
                reference_position=sample.reference_position,
                params=self._params,
                scene_complexity=self._compute_scene_complexity(sample),
                vehicle_density=self._compute_vehicle_density(sample),
                neighbor_density=self._compute_neighbor_density(sample),
                communication_history=self._load_communication_history(),
            )
            final_score = max(self._params.min_score_clip, min(self._params.max_score_clip, base_score))
            vehicle_scores[v.vehicle_id] = final_score
        
        # 점수 순으로 정렬
        sorted_vehicles = sorted(vehicle_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 해당 차량의 순위 찾기
        for rank, (vid, score) in enumerate(sorted_vehicles):
            if vid == vehicle_id:
                return rank
        
        return len(sample.vehicles)  # 찾지 못한 경우 마지막 순위
    
    def _generate_malicious_different_results(self, candidate_vehicle_id: str, peer_vehicle_id: str, sample: DatasetSample, base_result: list[dict]) -> list[dict]:
        """악성 에이전트가 협력자별로 미묘하게 다른 협업 인식 결과를 생성"""
        import random
        
        # 협력자별로 다른 시드 사용하여 미묘하게 다른 결과 생성
        random.seed(hash(candidate_vehicle_id + peer_vehicle_id + "malicious") % 1000)
        
        # 기본 결과를 기반으로 미묘한 조작 수행
        malicious_result = []
        
        if base_result:
            # 기존 결과의 70-80%는 유지하고 나머지만 조작
            keep_ratio = random.uniform(0.7, 0.8)
            num_to_keep = int(len(base_result) * keep_ratio)
            
            # 일부 결과는 그대로 유지
            kept_indices = random.sample(range(len(base_result)), min(num_to_keep, len(base_result)))
            for i in kept_indices:
                malicious_result.append(base_result[i].copy())
            
            # 나머지 결과는 미묘하게 조작
            for i, detection in enumerate(base_result):
                if i not in kept_indices:
                    # 미묘한 위치 조작 (5-15m 이동)
                    offset_x = random.uniform(-15, 15)
                    offset_y = random.uniform(-15, 15)
                    
                    # 미묘한 크기 조작 (10-20% 변화)
                    size_factor = random.uniform(0.8, 1.2)
                    
                    # 미묘한 신뢰도 조작 (0.1-0.3 변화)
                    confidence_offset = random.uniform(-0.3, 0.3)
                    
                    malicious_detection = {
                        "x1": detection["x1"] + offset_x,
                        "y1": detection["y1"] + offset_y,
                        "x2": detection["x2"] + offset_x + (detection["x2"] - detection["x1"]) * (size_factor - 1),
                        "y2": detection["y2"] + offset_y + (detection["y2"] - detection["y1"]) * (size_factor - 1),
                        "confidence": max(0.1, min(1.0, detection["confidence"] + confidence_offset)),
                        "class": detection["class"],
                        "source": f"malicious_subtle_from_{candidate_vehicle_id}_to_{peer_vehicle_id}"
                    }
                    malicious_result.append(malicious_detection)
        else:
            # 기본 결과가 없는 경우, 정상적인 위치 근처에 미묘한 조작 결과 생성
            num_detections = random.randint(1, 2)
            for i in range(num_detections):
                # 실제 환경 근처 위치에 생성 (더 현실적)
                x1 = random.uniform(-30, 30)
                y1 = random.uniform(-30, 30)
                x2 = x1 + random.uniform(8, 20)
                y2 = y1 + random.uniform(6, 15)
                confidence = random.uniform(0.4, 0.8)  # 적당한 신뢰도
                
                malicious_detection = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class": "car",
                    "source": f"malicious_subtle_from_{candidate_vehicle_id}_to_{peer_vehicle_id}"
                }
                malicious_result.append(malicious_detection)
        
        return malicious_result
    
    def _generate_malicious_random_results(self, vehicle_id: str, sample: DatasetSample) -> list[dict]:
        """
        악성 에이전트가 협력자 정보를 부분적으로 무시하고 생성한 조작된 협업 인식 결과
        
        이 함수가 생성한 조작 detection은:
        - 포함 증명(prove_inclusion)과 협력자 합의(peer_consensus_certification)에서 실제로 사용됨
        - IoU 비교를 통해 검증 실패를 유발
        - 그러나 ROBOSAC 실행에는 전달되지 않아 실제 협업 인식 성능에는 영향을 주지 않음
        - 목적: 보안 검증이 악성 차량을 탐지하고 거부할 수 있음을 시연
        """
        import random
        
        # 미묘한 조작 결과 생성
        random.seed(hash(vehicle_id + "random") % 1000)
        
        malicious_result = []
        
        # 실제 협력자들의 정보를 일부 활용하되 조작
        if sample.vehicles:
            # 협력자들의 평균 위치 계산
            avg_x = sum(v.position.x for v in sample.vehicles if v.vehicle_id != vehicle_id) / max(1, len(sample.vehicles) - 1)
            avg_y = sum(v.position.y for v in sample.vehicles if v.vehicle_id != vehicle_id) / max(1, len(sample.vehicles) - 1)
            
            # 평균 위치 근처에 일부는 정상, 일부는 조작된 결과 생성
            num_detections = random.randint(2, 4)
            
            for i in range(num_detections):
                if i < num_detections // 2:
                    # 절반은 정상적인 위치 근처에 생성 (협력자 정보 활용)
                    x1 = avg_x + random.uniform(-20, 20)
                    y1 = avg_y + random.uniform(-20, 20)
                    confidence = random.uniform(0.6, 0.9)  # 높은 신뢰도
                else:
                    # 나머지 절반은 조작된 위치에 생성
                    x1 = avg_x + random.uniform(-50, 50)
                    y1 = avg_y + random.uniform(-50, 50)
                    confidence = random.uniform(0.3, 0.6)  # 낮은 신뢰도
                
                x2 = x1 + random.uniform(8, 25)
                y2 = y1 + random.uniform(6, 18)
                
                malicious_detection = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class": "car",
                    "source": f"malicious_partial_from_{vehicle_id}"
                }
                malicious_result.append(malicious_detection)
        else:
            # 협력자가 없는 경우, 현실적인 위치에 생성
            num_detections = random.randint(1, 2)
            for i in range(num_detections):
                x1 = random.uniform(-40, 40)
                y1 = random.uniform(-40, 40)
                x2 = x1 + random.uniform(8, 20)
                y2 = y1 + random.uniform(6, 15)
                confidence = random.uniform(0.4, 0.7)
                
                malicious_detection = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class": "car",
                    "source": f"malicious_partial_from_{vehicle_id}"
                }
                malicious_result.append(malicious_detection)
        
        return malicious_result
    
    def _generate_malicious_subtle_results(self, vehicle_id: str, sample: DatasetSample) -> list[dict]:
        """
        미묘한 조작 공격: 정상 결과와 유사하지만 약간의 조작
        
        이 함수가 생성한 조작 detection은:
        - 포함 증명(prove_inclusion)과 협력자 합의(peer_consensus_certification)에서 실제로 사용됨
        - IoU 비교를 통해 검증 실패를 유발
        - 그러나 ROBOSAC 실행에는 전달되지 않아 실제 협업 인식 성능에는 영향을 주지 않음
        - 목적: 보안 검증이 악성 차량을 탐지하고 거부할 수 있음을 시연
        """
        import random
        
        # 시드 제거하여 더 다양한 공격 패턴 생성
        # random.seed(hash(vehicle_id + "subtle") % 1000)
        
        # 정상적인 협업 결과를 기반으로 미묘한 조작
        base_result = self._generate_fused_detections(sample)
        malicious_result = []
        
        if base_result:
            for detection in base_result:
                # 70% 확률로 정상 유지, 30% 확률로 미묘한 조작 (더 공격적으로)
                if random.random() < 0.7:
                    malicious_result.append(detection.copy())
                else:
                    # 미묘한 조작: 위치를 3-8m 이동 (더 큰 변화)
                    offset_x = random.uniform(-8, 8)
                    offset_y = random.uniform(-8, 8)
                    
                    # 신뢰도를 더 크게 조작
                    confidence_offset = random.uniform(-0.2, 0.2)
                    
                    malicious_detection = {
                        "x1": detection["x1"] + offset_x,
                        "y1": detection["y1"] + offset_y,
                        "x2": detection["x2"] + offset_x,
                        "y2": detection["y2"] + offset_y,
                        "confidence": max(0.1, min(1.0, detection["confidence"] + confidence_offset)),
                        "class": detection["class"],
                        "source": f"malicious_subtle_from_{vehicle_id}"
                    }
                    malicious_result.append(malicious_detection)
        else:
            # 기본 결과가 없는 경우, 현실적인 위치에 생성
            num_detections = random.randint(1, 3)  # 더 많은 검출 생성
            for i in range(num_detections):
                x1 = random.uniform(-30, 30)  # 더 넓은 범위
                y1 = random.uniform(-30, 30)
                x2 = x1 + random.uniform(6, 20)  # 더 다양한 크기
                y2 = y1 + random.uniform(4, 16)
                confidence = random.uniform(0.3, 0.9)  # 더 넓은 신뢰도 범위
                
                malicious_detection = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class": "car",
                    "source": f"malicious_subtle_from_{vehicle_id}"
                }
                malicious_result.append(malicious_detection)
        
        return malicious_result
    
    def _generate_malicious_adaptive_results(self, candidate_vehicle_id: str, peer_vehicle_id: str, sample: DatasetSample, base_result: list[dict]) -> list[dict]:
        """
        적응적 공격: 보안 검증을 우회하려는 시도
        
        이 함수가 생성한 조작 detection은:
        - 포함 증명(prove_inclusion)과 협력자 합의(peer_consensus_certification)에서 실제로 사용됨
        - IoU 비교를 통해 검증 실패를 유발
        - 그러나 ROBOSAC 실행에는 전달되지 않아 실제 협업 인식 성능에는 영향을 주지 않음
        - 목적: 보안 검증이 악성 차량을 탐지하고 거부할 수 있음을 시연
        """
        import random
        
        # 시드 제거하여 더 다양한 공격 패턴 생성
        # random.seed(hash(candidate_vehicle_id + peer_vehicle_id + "adaptive") % 1000)
        
        malicious_result = []
        
        if base_result:
            # 기본 결과의 일부만 유지하되, 임계값 근처에서 실패하도록 조작
            keep_ratio = random.uniform(0.4, 0.6)  # 40-60%만 유지 (더 공격적으로)
            num_to_keep = int(len(base_result) * keep_ratio)
            
            # 일부 결과는 그대로 유지
            kept_indices = random.sample(range(len(base_result)), min(num_to_keep, len(base_result)))
            for i in kept_indices:
                malicious_result.append(base_result[i].copy())
            
            # 나머지 결과는 임계값 근처에서 실패하도록 조작
            for i, detection in enumerate(base_result):
                if i not in kept_indices:
                    # 임계값 근처에서 실패하도록 조작 (IoU 0.2-0.4 범위로 더 공격적)
                    offset_x = random.uniform(-12, 12)  # 더 큰 이동
                    offset_y = random.uniform(-12, 12)
                    
                    malicious_detection = {
                        "x1": detection["x1"] + offset_x,
                        "y1": detection["y1"] + offset_y,
                        "x2": detection["x2"] + offset_x,
                        "y2": detection["y2"] + offset_y,
                        "confidence": detection["confidence"] * random.uniform(0.6, 0.8),  # 더 낮춤
                        "class": detection["class"],
                        "source": f"malicious_adaptive_from_{candidate_vehicle_id}_to_{peer_vehicle_id}"
                    }
                    malicious_result.append(malicious_detection)
        else:
            # 기본 결과가 없는 경우, 임계값 근처에서 실패하도록 생성
            num_detections = random.randint(1, 3)  # 더 많은 검출 생성
            for i in range(num_detections):
                x1 = random.uniform(-25, 25)  # 더 넓은 범위
                y1 = random.uniform(-25, 25)
                x2 = x1 + random.uniform(4, 15)  # 더 다양한 크기
                y2 = y1 + random.uniform(3, 12)
                confidence = random.uniform(0.2, 0.4)  # 더 낮은 신뢰도
                
                malicious_detection = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                    "class": "car",
                    "source": f"malicious_adaptive_from_{candidate_vehicle_id}_to_{peer_vehicle_id}"
                }
                malicious_result.append(malicious_detection)
        
        return malicious_result


