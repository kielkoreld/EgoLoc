#!/bin/bash
# EgoLoc 환경 자동 설치 스크립트

set -e  # 오류 발생 시 중단

echo "=== EgoLoc 환경 설치 시작 ==="

# 1. conda 환경 생성
echo "[1/3] Conda 환경 생성 중..."
conda env create -f EgoLoc.yaml

# 2. 환경 활성화
echo "[2/3] 환경 활성화 중..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate EgoLoc

# CUDA 라이브러리 경로 설정
CONDA_PREFIX=$(conda info --base)/envs/EgoLoc
if [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    echo "CUDA 라이브러리 경로 설정: $CONDA_PREFIX/lib"
fi

# 3. nuscenes-devkit 설치 (의존성 체크 없이)
echo "[3/3] nuscenes-devkit 설치 중 (의존성 체크 없이)..."
pip install --no-deps nuscenes-devkit==1.0.9

echo ""
echo "=== 설치 완료! ==="
echo "환경을 활성화하려면: conda activate EgoLoc"

