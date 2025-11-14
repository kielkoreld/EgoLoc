from coperception.models.det.base.IntermediateModelBase import IntermediateModelBase
import torch
import os
import time

# Lazy import for EgoLoc fusion handler (only imported when needed)
_apply_egoloc_fusion_noise = None

def _get_egoloc_fusion_handler():
    """Lazy import helper for EgoLoc fusion handler"""
    global _apply_egoloc_fusion_noise
    if _apply_egoloc_fusion_noise is None:
        try:
            from .egoloc_fusion_handler import apply_egoloc_fusion_noise
            _apply_egoloc_fusion_noise = apply_egoloc_fusion_noise
        except ImportError:
            _apply_egoloc_fusion_noise = False  # Mark as unavailable
    return _apply_egoloc_fusion_noise

class FusionBase(IntermediateModelBase):
    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        num_agent=5,
        compress_level=0,
        only_v2i=False,
    ):
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level, only_v2i)
        self.num_agent = 0

    def fusion(self):
        raise NotImplementedError(
            "Please implement this method for specific fusion strategies"
        )

    def forward(self, bevs, trans_matrices, num_agent_tensor, batch_size=1, pert=None, eps=None, attacker_list=None, ego_agent=None, unadv_pert=None, kick=False, no_fuse=False, collab_agent_list=None, trial_agent_id=None):

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        encoded_layers = self.u_encoder(bevs)
        device = bevs.device


        if not no_fuse:
            # print("Fusion Activated")

            feat_maps, size = super().get_feature_maps_and_size(encoded_layers)

            # print("Feature Maps Shape: ", feat_maps.shape)

            # if pert is not None:
                # print("Perturbation is applied on agent {}".format(attacker_list))
                # clip
                # eta = torch.clamp(pert, min=-eps, max=eps)
                # Apply perturbation
                # feat_maps[attacker_list] = feat_maps[attacker_list] + eta
            # else:
            #     print("Perturbation is not applied")
                                    
            # if unadv_pert is not None:
            #     # print("Unadversarial perturbation is applied on agent 2")
            #     feat_maps[2] = feat_maps[2] + unadv_pert
            # else:
                # print("Unadversarial perturbation is not applied")

            feat_list = super().build_feature_list(batch_size, feat_maps)

            local_com_mat = super().build_local_communication_matrix(
                feat_list
            )  # [2 5 512 16 16] [batch, agent, channel, height, width]

            
            local_com_mat_update = super().build_local_communication_matrix(
                feat_list
            )  # to avoid the inplace operation

            

            for b in range(batch_size):

                self.num_agent = num_agent_tensor[b, 0]
                
                for i in range(self.num_agent):
                    self.tg_agent = local_com_mat[b, i]
                    
                    # EgoLoc 공격 시나리오 최적화 중에만 pert 적용
                    # 원본과 동일: FusionBase에서는 직접 pert를 적용하지 않음
                    # pert는 DetModelBase의 build_neighbors_feature_list에서만 적용됨
                    # 단, EgoLoc 공격 시나리오에서 최적화 중(gradient 계산 중)인 경우에만
                    # ego agent feature에 pert를 적용하여 gradient 계산 가능하도록 함
                    # (ego agent는 자신의 neighbor가 아니므로 DetModelBase에서 pert가 적용되지 않음)
                    # 비공격 시나리오에서는 pert=None이므로 이 체크는 실행되지 않아 원본과 동일한 성능 유지
                    # 최적화: 가장 빠르게 False가 되는 조건을 먼저 체크 (i == ego_agent가 대부분 False)
                    if (ego_agent is not None and i == ego_agent 
                        and pert is not None and attacker_list is not None and i in attacker_list 
                        and eps is not None and pert.requires_grad):
                        # EgoLoc 공격 시나리오 최적화 중: ego agent feature에 pert 적용 (gradient 계산용)
                        eta = torch.clamp(pert[i], min=-eps, max=eps)
                        self.tg_agent = self.tg_agent + eta
                    
                    self.neighbor_feat_list = []
                    self.neighbor_feat_list.append(self.tg_agent)
                    all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]
                    # i == ego_agent: ego agent
                    if ego_agent is not None and i == ego_agent:       
                        super().build_neighbors_feature_list(
                            b,
                            i,
                            all_warp,
                            self.num_agent,
                            local_com_mat,
                            device,
                            size,
                            trans_matrices,
                            collab_agent_list, 
                            trial_agent_id,
                            pert,
                            attacker_list,
                            eps
                        )
                    else:
                        super().build_neighbors_feature_list(
                            b,
                            i,
                            all_warp,
                            self.num_agent,
                            local_com_mat,
                            device,
                            size,
                            trans_matrices,
                        )

                    # feature update
                    local_com_mat_update[b, i] = self.fusion()

            # weighted feature maps is passed to decoder
            feat_fuse_mat = super().agents_to_batch(local_com_mat_update)
            
            # EgoLoc 공격 시나리오: fusion result에 노이즈 추가
            # 검증 단계 통과 여부 확인:
            # - Baseline: ego_agent == 0이면 항상 공격자이므로 노이즈 추가
            # - EgoLoc: ego_agent in attacker_list이면 검증을 통과한 공격자이므로 노이즈 추가
            # - EgoLoc: ego_agent not in attacker_list이면 검증을 통과하지 못한 것 (정상) → 노이즈 추가 안 함
            # 기존 ROBOSAC (baseline)에서는 pert=None이거나 ego_agent가 attacker_list에 없으므로 스킵됨
            # 최적화: 조건 체크 순서 최적화 (가장 빠르게 False가 되는 조건을 먼저 체크)
            should_apply_noise = False
            if pert is not None and ego_agent is not None and attacker_list is not None:
                # Baseline과 EgoLoc 구분: EGOL0C_ATTACKER_LIST 환경 변수 존재 여부로 판단
                # - EGOL0C_ATTACKER_LIST가 있으면 EgoLoc 실행
                # - EGOL0C_ATTACKER_LIST가 없고 EGOL0C_BASELINE_ATTACKER_LIST가 있으면 Baseline 실행
                egoloc_attacker_list_env = os.environ.get("EGOL0C_ATTACKER_LIST", "").strip()
                baseline_attacker_list_env = os.environ.get("EGOL0C_BASELINE_ATTACKER_LIST", "").strip()
                is_baseline = (not egoloc_attacker_list_env) and baseline_attacker_list_env
                
                if is_baseline:
                    # Baseline: ego_agent가 항상 공격자이므로 노이즈 추가
                    # baseline_attacker_list_env에서 읽은 공격자 리스트에 ego_agent가 포함되어 있으면 노이즈 추가
                    if baseline_attacker_list_env:
                        try:
                            baseline_attacker_list = [int(x.strip()) for x in baseline_attacker_list_env.split(",") if x.strip()]
                            if ego_agent in baseline_attacker_list:
                                should_apply_noise = True
                        except Exception:
                            pass
                else:
                    # EgoLoc: 검증을 통과한 공격자만 노이즈 추가 (ego_agent in attacker_list)
                    # 후보 큐의 1순위가 공격자이지만, 검증을 통과하지 못하면 attacker_list에 포함되지 않음
                    if ego_agent in attacker_list:
                        should_apply_noise = True
            
            if should_apply_noise:
                # Lazy import를 사용하여 필요할 때만 import (첫 호출 시에만 실제 import 수행)
                apply_egoloc_fusion_noise = _get_egoloc_fusion_handler()
                if apply_egoloc_fusion_noise and not (hasattr(pert, 'requires_grad') and pert.requires_grad):
                    # Baseline: agent0이 항상 공격자이므로 노이즈 추가
                    # EgoLoc: 검증을 통과한 공격자만 노이즈 추가 (ego_agent in attacker_list)
                    # pert가 detach된 경우(requires_grad=False)에만 fusion result에 노이즈 추가
                    feat_fuse_mat = apply_egoloc_fusion_noise(
                        feat_fuse_mat, pert, attacker_list, ego_agent, eps, batch_size
                    )

        else:
            # print("Fusion disabled")
            feat_maps, size = super().get_feature_maps_and_size(encoded_layers)

            feat_list = super().build_feature_list(batch_size, feat_maps)

            local_com_mat = super().build_local_communication_matrix(
                feat_list
            )  # [2 5 512 16 16] [batch, agent, channel, height, width]
            feat_fuse_mat = super().agents_to_batch(local_com_mat)
            
            # no_fuse=True일 때는 apply_egoloc_fusion_noise를 호출하지 않음
            # (기존 ROBOSAC과 동일한 동작)

        decoded_layers = super().get_decoded_layers(
            encoded_layers, feat_fuse_mat, batch_size
        )
        x = decoded_layers[0]

        cls_preds, loc_preds, result = super().get_cls_loc_result(x)

        if self.kd_flag == 1:
            return (result, *decoded_layers, feat_fuse_mat)
        else:
            return result
