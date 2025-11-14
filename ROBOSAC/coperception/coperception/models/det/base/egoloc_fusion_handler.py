"""
EgoLoc Fusion 결과에 노이즈 추가 모듈

이 모듈은 ROBOSAC의 원본 코드를 수정하지 않고 EgoLoc 공격 시나리오의 fusion result에 노이즈를 추가합니다.
"""

import torch


def apply_egoloc_fusion_noise(feat_fuse_mat, pert, attacker_list, ego_agent, eps, batch_size):
    """
    EgoLoc 공격 시나리오에서 최적화된 pert를 fusion result에 추가합니다.
    
    Args:
        feat_fuse_mat: Fusion 결과 feature map [batch*agent, channel, height, width]
        pert: 최적화된 perturbation 텐서 [num_agent, channel, height, width]
        attacker_list: 공격자 리스트
        ego_agent: Ego agent 인덱스
        eps: Epsilon 값
        batch_size: 배치 크기
        
    Returns:
        feat_fuse_mat: 노이즈가 추가된 fusion result
    """
    # EgoLoc 공격 시나리오 확인: pert가 detach된 경우(requires_grad=False) fusion result에 추가
    should_add_noise_to_fusion = (pert is not None and attacker_list is not None 
                                 and ego_agent is not None and eps is not None
                                 and (ego_agent in attacker_list if attacker_list is not None else False))
    
    if not should_add_noise_to_fusion:
        return feat_fuse_mat
    
    # pert가 requires_grad=False인 경우 (EgoLoc 공격 시나리오: 최적화 완료 후)
    if not (hasattr(pert, 'requires_grad') and pert.requires_grad):
        # EgoLoc 공격 시나리오: 최적화된 pert를 fusion result에 추가
        num_agent = feat_fuse_mat.shape[0] // batch_size if batch_size > 0 else 1
        for b in range(batch_size):
            ego_idx_in_batch = b * num_agent + ego_agent
            if ego_idx_in_batch < feat_fuse_mat.shape[0] and pert.shape[0] > ego_agent:
                # 최적화된 pert에서 ego agent 부분 사용
                # feat_fuse_mat의 shape: [batch*agent, channel, height, width]
                # pert[ego_agent]의 shape: [channel, height, width] (256, 32, 32)
                # feat_fuse_mat[ego_idx_in_batch]의 shape과 맞추기 위해 resize 필요할 수 있음
                pert_ego = pert[ego_agent]  # [256, 32, 32]
                feat_shape = feat_fuse_mat[ego_idx_in_batch].shape  # [C, H, W]
                
                # shape이 맞으면 직접 사용, 아니면 resize
                if pert_ego.shape == feat_shape:
                    noise = torch.clamp(pert_ego, min=-eps, max=eps)
                else:
                    # shape이 다르면 interpolation 또는 slice
                    noise = torch.clamp(pert_ego[:feat_shape[0], :feat_shape[1], :feat_shape[2]], min=-eps, max=eps)
                    if noise.shape != feat_shape:
                        # 최종적으로 shape이 안 맞으면 랜덤 노이즈 사용
                        noise = torch.randn_like(feat_fuse_mat[ego_idx_in_batch]) * eps
                
                # Fusion result에 최적화된 pert 추가 (ego가 공격자이므로 최종 결과에 노이즈 전파)
                feat_fuse_mat[ego_idx_in_batch] = feat_fuse_mat[ego_idx_in_batch] + noise
    
    return feat_fuse_mat

