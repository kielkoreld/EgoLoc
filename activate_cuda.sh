#!/bin/bash
# CUDA 라이브러리 경로 설정 스크립트
# 환경 활성화 후 이 스크립트를 source하거나 실행하세요

if [ -z "$CONDA_PREFIX" ]; then
    echo "오류: conda 환경이 활성화되지 않았습니다."
    echo "먼저 'conda activate EgoLoc'를 실행하세요."
    exit 1
fi

# conda 환경의 CUDA 라이브러리 경로 추가
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# WSL에서 CUDA 라이브러리 경로 추가 (있는 경우)
if [ -d "/usr/lib/wsl/lib" ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

# 시스템 CUDA 경로 추가 (있는 경우)
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

echo "CUDA 라이브러리 경로 설정 완료:"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

