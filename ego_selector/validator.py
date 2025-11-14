from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass(frozen=True)
class ProofResult:
    """포함 증명 결과

    - is_included: 협력 차량의 기여 정보가 협업 인식 결과에 충분히 반영되었는지 여부
    - match_ratio: IoU 기준으로 매칭된 비율(기여 박스 대비)
    - total_contributed / total_matched: 기여 박스 수 / 매칭된 박스 수
    - details: 임계값 등 부가 정보
    """
    is_included: bool
    match_ratio: float
    total_contributed: int
    total_matched: int
    details: Dict[str, float]


def _to_xyxy(box: Dict) -> Optional[Tuple[float, float, float, float]]:
    """박스 딕셔너리를 (x1,y1,x2,y2)로 정규화. 변환 불가 시 None."""
    # Support either (x1,y1,x2,y2) or (x,y,w,h)
    if all(k in box for k in ("x1", "y1", "x2", "y2")):
        x1 = float(box["x1"]) ; y1 = float(box["y1"]) ; x2 = float(box["x2"]) ; y2 = float(box["y2"]) 
        if x2 < x1 or y2 < y1:
            return None
        return (x1, y1, x2, y2)
    if all(k in box for k in ("x", "y", "w", "h")):
        x = float(box["x"]) ; y = float(box["y"]) ; w = float(box["w"]) ; h = float(box["h"]) 
        if w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)
    return None


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """두 박스의 IoU 계산"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aarea = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    barea = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = aarea + barea - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_boxes(contrib_xyxy: List[Tuple[float, float, float, float]], fused_xyxy: List[Tuple[float, float, float, float]], iou_thr: float) -> Tuple[int, int]:
    """일대일 매칭(탐욕적)으로 매칭 수/기여 수 반환"""
    matched = 0
    used = [False] * len(fused_xyxy)
    for a in contrib_xyxy:
        for j, b in enumerate(fused_xyxy):
            if used[j]:
                continue
            if _iou(a, b) >= iou_thr:
                used[j] = True
                matched += 1
                break
    return matched, len(contrib_xyxy)


def prove_inclusion(contributed_detections: List[Dict], fused_detections: List[Dict], iou_threshold: float = 0.5, min_match_ratio: float = 0.2) -> ProofResult:
    """포함 증명: 협력 차량이 기여한 검출 결과가 협업 인식 결과에 충분히 반영되었는지 검증"""
    contrib_xyxy = [b for b in (_to_xyxy(x) for x in contributed_detections) if b is not None]
    fused_xyxy = [b for b in (_to_xyxy(x) for x in fused_detections) if b is not None]
    if len(contrib_xyxy) == 0:
        return ProofResult(is_included=False, match_ratio=0.0, total_contributed=0, total_matched=0, details={"reason": 0.0})
    matched, total = _match_boxes(contrib_xyxy, fused_xyxy, iou_threshold)
    ratio = 0.0 if total == 0 else (matched / total)
    return ProofResult(
        is_included=ratio >= min_match_ratio,
        match_ratio=ratio,
        total_contributed=total,
        total_matched=matched,
        details={"iou_threshold": iou_threshold, "min_match_ratio": min_match_ratio},
    )


# -------- Peer-only consensus (excluding current ego candidate) --------

@dataclass(frozen=True)
class PeerConsensus:
    """협력자 합의 결과(유력 자아 후보 제외)"""
    consensus: bool
    num_certified: int
    total_peers: int
    per_peer_unmatched: Dict[str, float]
    threshold: float


def peer_consensus_certification(
    peers_results: Dict[str, List[Dict]],
    exclude_agent_id: Optional[str],
    iou_threshold: float = 0.5,
    max_unmatched_ratio: float = 0.5,
    quorum_ratio: float = 0.6,
) -> PeerConsensus:
    """협력자 합의 검증(유력 자아 후보 제외)"""
    # Build normalized boxes
    xyxy_by_peer: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for aid, boxes in peers_results.items():
        if exclude_agent_id is not None and aid == exclude_agent_id:
            continue
        xyxy_by_peer[aid] = [b for b in (_to_xyxy(x) for x in boxes) if b is not None]

    peer_ids = list(xyxy_by_peer.keys())
    per_unmatched: Dict[str, float] = {}

    def unmatched_ratio(a: List[Tuple[float, float, float, float]], b: List[Tuple[float, float, float, float]]) -> float:
        """a가 b와 얼마나 불일치하는지(매칭 실패 비율)를 계산"""
        if not a:
            return 0.0
        used = [False] * len(b)
        matched = 0
        for aa in a:
            for j, bb in enumerate(b):
                if used[j]:
                    continue
                if _iou(aa, bb) >= iou_threshold:
                    used[j] = True
                    matched += 1
                    break
        return 1.0 - (matched / len(a))

    # For each peer, compare to the concatenation (union-approx) of all other peers
    for aid in peer_ids:
        a = xyxy_by_peer[aid]
        others: List[Tuple[float, float, float, float]] = []
        for bid in peer_ids:
            if bid == aid:
                continue
            others.extend(xyxy_by_peer.get(bid, []))
        per_unmatched[aid] = unmatched_ratio(a, others)

    certified = [aid for aid, ratio in per_unmatched.items() if ratio <= max_unmatched_ratio]
    total = len(peer_ids)
    need = int(max(1, round(total * quorum_ratio)))
    return PeerConsensus(
        consensus=len(certified) >= need,
        num_certified=len(certified),
        total_peers=total,
        per_peer_unmatched=per_unmatched,
        threshold=max_unmatched_ratio,
    )

