"""
EgoLoc 공격 시나리오 처리 모듈

이 모듈은 ROBOSAC의 원본 코드를 수정하지 않고 EgoLoc 공격 시나리오를 처리합니다.
"""

import torch
import random
import os
import time


def handle_egoloc_attack_scenario(args, data, num_sensor, ego_idx, all_agent_list, device, 
                                  fafmodule, batch_size, pseudo_gt, print_and_write_log,
                                  robosac_attacker_list=None):
    """
    EgoLoc 공격 시나리오를 처리합니다.
    
    Args:
        args: ROBOSAC arguments 객체
        data: 입력 데이터 딕셔너리
        num_sensor: 센서 수
        ego_idx: Ego agent 인덱스
        all_agent_list: 모든 agent 리스트
        device: PyTorch device
        fafmodule: FAF 모듈
        batch_size: 배치 크기
        pseudo_gt: Pseudo ground truth
        print_and_write_log: 로그 출력 함수
        robosac_attacker_list: 기존 ROBOSAC의 attacker_list (선택적, None이면 기존 로직 사용)
        
    Returns:
        pert: 생성된 perturbation 텐서
        attacker_list: 공격자 리스트 (ROBOSAC의 공격자 + ego_agent)
    """
    # EgoLoc 공격 시나리오 확인
    has_egoloc_attack_mode = hasattr(args, 'egoloc_attack_mode')
    egoloc_attack_mode_value = getattr(args, 'egoloc_attack_mode', None)
    egoloc_attack_active = has_egoloc_attack_mode and args.egoloc_attack_mode
    
    if not egoloc_attack_active:
        return None, None
    
    # 검증 단계 이전에 생성된 공격자 리스트 확인 (환경 변수에서 읽기)
    # Baseline 실행 시: EGOL0C_BASELINE_ATTACKER_LIST 사용 (agent0이 항상 공격자)
    # EgoLoc 실행 시: EGOL0C_ATTACKER_LIST 사용 (검증 전 큐의 1순위가 공격자)
    # 구분 방법: EGOL0C_ATTACKER_LIST가 있으면 EgoLoc 실행, 없으면 Baseline 실행
    egoloc_attacker_list = []
    
    # EgoLoc 실행 여부 확인: EGOL0C_ATTACKER_LIST 환경 변수 존재 여부로 판단
    egoloc_attacker_list_env = os.environ.get("EGOL0C_ATTACKER_LIST", "").strip()
    baseline_attacker_list_env = os.environ.get("EGOL0C_BASELINE_ATTACKER_LIST", "").strip()
    
    if egoloc_attacker_list_env:
        # EgoLoc 실행: EGOL0C_ATTACKER_LIST 사용 (검증 전 큐의 1순위가 공격자)
        try:
            # 환경 변수에서 공격자 리스트 파싱 (예: "0,1,2")
            egoloc_attacker_list = [int(x.strip()) for x in egoloc_attacker_list_env.split(",") if x.strip()]
            print_and_write_log(f"[EgoLoc Attack] EgoLoc 실행: 검증 전 공격자 리스트 (환경 변수): {egoloc_attacker_list}")
        except Exception as e:
            print_and_write_log(f"[EgoLoc Attack] EgoLoc 환경 변수 파싱 실패: {e}")
            egoloc_attacker_list = []
    elif baseline_attacker_list_env:
        # Baseline 실행: EGOL0C_BASELINE_ATTACKER_LIST 사용 (ego_agent가 항상 공격자)
        try:
            egoloc_attacker_list = [int(x.strip()) for x in baseline_attacker_list_env.split(",") if x.strip()]
            print_and_write_log(f"[EgoLoc Attack] Baseline 실행: Baseline 공격자 리스트 (환경 변수): {egoloc_attacker_list}")
        except Exception as e:
            print_and_write_log(f"[EgoLoc Attack] Baseline 환경 변수 파싱 실패: {e}")
            egoloc_attacker_list = []  # 파싱 실패 시 빈 리스트
    else:
        # 환경 변수가 없는 경우: 공격자 리스트 없음
        egoloc_attacker_list = []
        print_and_write_log(f"[EgoLoc Attack] 환경 변수 없음: 공격자 리스트 없음")
    
    # EgoLoc 공격 시나리오: ego가 공격자인 경우 처리
    ego_is_attacker = False
    # Baseline과 EgoLoc 구분: EGOL0C_ATTACKER_LIST 환경 변수 존재 여부로 판단
    egoloc_attacker_list_env = os.environ.get("EGOL0C_ATTACKER_LIST", "").strip()
    baseline_attacker_list_env = os.environ.get("EGOL0C_BASELINE_ATTACKER_LIST", "").strip()
    is_baseline = (not egoloc_attacker_list_env) and baseline_attacker_list_env
    
    if is_baseline:
        # Baseline: ego_agent가 항상 공격자 (baseline_attacker_list_env에서 읽어옴)
        if baseline_attacker_list_env:
            try:
                baseline_attacker_list = [int(x.strip()) for x in baseline_attacker_list_env.split(",") if x.strip()]
                if ego_idx in baseline_attacker_list:
                    ego_is_attacker = True
            except Exception:
                pass
    elif hasattr(args, 'egoloc_no_defense') and args.egoloc_no_defense:
        # EgoLoc No-defense: 검증 없이 1위 선택, 1위가 공격자이므로 ego가 공격자
        ego_is_attacker = True
    elif egoloc_attacker_list and ego_idx in egoloc_attacker_list:
        # EgoLoc Defense: 검증 전 공격자 리스트에 ego가 포함되어 있으면 공격자
        # 검증 단계를 통과했는지는 나중에 확인 (FusionBase.py에서)
        ego_is_attacker = True
    
    # 공격자 리스트 생성: 기존 ROBOSAC의 attacker_list에 EgoLoc 공격자 추가
    # 목표: ROBOSAC의 공격자 1명 + EgoLoc 공격자 = 총 2명의 공격자
    # - Baseline: ROBOSAC 공격자 + ego_agent (agent1)
    # - EgoLoc: ROBOSAC 공격자 + 검증 전 큐의 1순위 차량 (egoloc_attacker_list)
    if robosac_attacker_list is not None:
        # 기존 ROBOSAC의 attacker_list가 있는 경우
        print_and_write_log(f"[EgoLoc Attack] 받은 기존 ROBOSAC 공격자 리스트: {robosac_attacker_list}")
        attacker_list = robosac_attacker_list.copy()
        
        # Baseline과 EgoLoc 구분
        if is_baseline:
            # Baseline: ego_agent를 추가
            print_and_write_log(f"[EgoLoc Attack] Baseline: 추가할 공격자 (ego_agent): {ego_idx}")
            if ego_idx not in attacker_list:
                attacker_list.append(ego_idx)
                print_and_write_log(f"[EgoLoc Attack] Baseline: 기존 ROBOSAC 공격자 {robosac_attacker_list} + ego_agent {ego_idx} = 최종 공격자 리스트: {attacker_list}")
            else:
                print_and_write_log(f"[EgoLoc Attack] Baseline: 기존 ROBOSAC 공격자 {robosac_attacker_list}에 ego_agent {ego_idx}가 이미 포함됨 → 최종 공격자 리스트: {attacker_list}")
        else:
            # EgoLoc: 검증 전 큐의 1순위 차량(egoloc_attacker_list)을 추가
            print_and_write_log(f"[EgoLoc Attack] EgoLoc: 추가할 공격자 (검증 전 1순위): {egoloc_attacker_list}")
            if egoloc_attacker_list:
                for attacker in egoloc_attacker_list:
                    if attacker not in attacker_list:
                        attacker_list.append(attacker)
                print_and_write_log(f"[EgoLoc Attack] EgoLoc: 기존 ROBOSAC 공격자 {robosac_attacker_list} + 검증 전 1순위 {egoloc_attacker_list} = 최종 공격자 리스트: {attacker_list}")
            else:
                # egoloc_attacker_list가 없는 경우 (예외 상황)
                print_and_write_log(f"[EgoLoc Attack] EgoLoc: egoloc_attacker_list가 없어 기존 ROBOSAC 공격자 리스트만 사용: {attacker_list}")
    elif ego_is_attacker:
        # 기존 ROBOSAC의 attacker_list가 없고, ego가 공격자인 경우
        # 검증 전 공격자 리스트가 있으면 사용, 없으면 ego를 포함하여 생성
        if egoloc_attacker_list:
            # 검증 전 공격자 리스트 사용 (ego가 포함되어 있음)
            attacker_list = egoloc_attacker_list.copy()
            # ego가 리스트에 없으면 추가
            if ego_idx not in attacker_list:
                attacker_list.append(ego_idx)
        else:
            # 검증 전 공격자 리스트가 없으면 기존 로직 사용
            other_agents = [i for i in all_agent_list if i != ego_idx]
            remaining_attackers = max(0, args.number_of_attackers - 1)
            if remaining_attackers > 0 and len(other_agents) > 0:
                other_attackers = random.sample(other_agents, k=min(remaining_attackers, len(other_agents)))
                attacker_list = [ego_idx] + other_attackers
            else:
                attacker_list = [ego_idx]
        print_and_write_log(f"[EgoLoc Attack] Ego agent {ego_idx} is attacker. Attacker list: {attacker_list}")
    else:
        # 기존 ROBOSAC의 attacker_list가 없고, ego가 공격자가 아닌 경우
        # 검증 전 공격자 리스트가 있으면 사용, 없으면 기존 로직 사용
        if egoloc_attacker_list:
            attacker_list = egoloc_attacker_list.copy()
        else:
            all_agent_list_without_ego = all_agent_list.copy()
            all_agent_list_without_ego.remove(ego_idx)
            if args.robosac == 'fix_attackers':
                attacker_list = random.sample(all_agent_list_without_ego, k=args.number_of_attackers)
            else:
                attacker_list = random.sample(all_agent_list_without_ego, k=args.number_of_attackers)
    
    # EgoLoc 공격 시나리오: 기존 ROBOSAC과 동일하게 최적화 루프 수행
    # 기존 ROBOSAC과 동일하게 초기화 (device 지정 없이, 나중에 .to(device) 사용)
    if args.adv_method == 'pgd':
        # PGD random init   
        pert = torch.randn(num_sensor, 256, 32, 32) * 0.1
    elif args.adv_method == 'bim' or args.adv_method == 'cw-l2':
        # BIM/CW-L2 zero init
        pert = torch.zeros(num_sensor, 256, 32, 32)
    else:
        raise NotImplementedError
    
    data['attacker_list'] = attacker_list
    data['eps'] = args.eps
    data['no_fuse'] = False
    
    # 공격 반복: perturbation 생성 및 최적화 (기존 ROBOSAC과 완전히 동일)
    # adv_iter 기본값 확인 및 설정
    adv_iter = getattr(args, 'adv_iter', 15)  # 기본값 15 (기존 ROBOSAC과 동일)
    if adv_iter <= 0:
        print_and_write_log(f"[EgoLoc Attack] Warning: adv_iter={adv_iter} is invalid, using default 15")
        adv_iter = 15
    
    for i in range(adv_iter):
        # 매 반복마다 requires_grad를 True로 설정 (기존 ROBOSAC과 동일)
        pert.requires_grad = True
        # Introduce adv perturbation
        data['pert'] = pert.to(device)
                
        # STEP 3: Use inverted classification ground truth, minimize loss wrt inverted gt, to generate adv attacks based on cls(only)
        # NOTE: cls_step 내부에서 backward(retain_graph=True)를 호출하여 pert에 대한 gradient도 계산됨
        cls_loss = fafmodule.cls_step(data, batch_size, ego_loss_only=args.ego_loss_only, 
                                     ego_agent=args.ego_agent, invert_gt=True, 
                                     self_result=pseudo_gt, adv_method=args.adv_method)

        # Gradient 확인 및 업데이트 (기존 ROBOSAC과 완전히 동일한 방식)
        # 기존 ROBOSAC 코드와 동일하게 pert.grad 체크 없이 직접 사용
        # (cls_step에서 gradient가 계산되므로 항상 존재해야 함)
        pert_alpha = getattr(args, 'pert_alpha', 0.1)  # 기본값 0.1 (기존 ROBOSAC과 동일)
        
        if pert.grad is None:
            print_and_write_log(f"[EgoLoc Attack] Warning: pert.grad is None at iteration {i + 1}, skipping update")
        else:
            pert = pert + pert_alpha * pert.grad.sign() * -1
        
        pert.detach_()
    
    # Detach and clone perturbations from Pytorch computation graph, in case of gradient misuse.
    pert = pert.detach().clone()
    # EgoLoc 공격 시나리오: 최적화 완료 후, fusion result에 노이즈 추가를 위해 pert를 detach
    # FusionBase.py에서 requires_grad=False인 경우 fusion result에 노이즈 추가
    data['attacker_list'] = attacker_list
    data['eps'] = args.eps
    data['no_fuse'] = False
    data['pert'] = pert.to(device)  # requires_grad=False로 detach된 상태
    
    # Baseline과 EgoLoc 구분하여 공격자 리스트 출력
    if is_baseline:
        print_and_write_log(f"[Baseline Attack] 공격자 리스트: {attacker_list}")
    else:
        print_and_write_log(f"[EgoLoc Attack] 공격자 리스트: {attacker_list}")
    
    return pert, attacker_list

