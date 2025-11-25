#!/bin/bash
# CUDA 라이브러리 경로를 conda 환경 활성화 시 자동으로 설정하는 스크립트
# 이 스크립트를 한 번 실행하면 환경 활성화 시마다 자동으로 설정됩니다

CONDA_BASE=$(conda info --base)
ENV_NAME="EgoLoc"
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"

if [ ! -d "$ENV_DIR" ]; then
    echo "오류: EgoLoc 환경이 설치되지 않았습니다."
    echo "먼저 'conda env create -f EgoLoc.yaml'를 실행하세요."
    exit 1
fi

# activate 스크립트 생성
ACTIVATE_SCRIPT="$ENV_DIR/etc/conda/activate.d/cuda_path.sh"
mkdir -p "$(dirname "$ACTIVATE_SCRIPT")"

cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# CUDA 라이브러리 경로 자동 설정

# conda 환경의 CUDA 라이브러리 경로 추가
if [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# WSL에서 CUDA 라이브러리 경로 추가 (있는 경우)
if [ -d "/usr/lib/wsl/lib" ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

# 시스템 CUDA 경로 추가 (있는 경우)
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi
EOF

chmod +x "$ACTIVATE_SCRIPT"
echo "CUDA 라이브러리 경로 자동 설정이 완료되었습니다!"
echo "이제 'conda activate EgoLoc'를 실행하면 자동으로 설정됩니다."

