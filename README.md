# EgoLoc(Ego-vehicle based on Location)

<img src="EgoLoc.png">


### 파일 구조

```
ego_selector/
  __init__.py
  types.py
  scoring.py
  priority_queue.py
  validator.py
  selector.py
  adapters/
    __init__.py
    dataset_adapter.py

examples/
  select_ego.py

ROBOSAC/
  ...

EgoLoc.yaml
```

### 빠른 시작
```bash
# 기본 JSON 샘플 예제 (내장 샘플 파일 사용)
python -m examples.select_ego

# V2X-Sim에서 선택 실행 (기본 경로/파라미터)
python -m examples.select_ego --mode v2x-sim

# 공격 시나리오
python -m examples.select_ego --mode v2x-sim --attack {subtle, adaptive}

#평가
... --robosac_mAP 추가
```

### 환경 설치

```bash
# 자동 설치 (권장)
chmod +x setup_egoloc.sh
./setup_egoloc.sh
conda activate EgoLoc
pip install --no-deps nuscenes-devkit==1.0.9
```

### CUDA 라이브러리 경로 설정 (WSL 환경)

WSL에서 CUDA 라이브러리 오류가 발생하는 경우:

```bash
# 환경 활성화 후
conda activate EgoLoc

# CUDA 라이브러리 경로 설정
source activate_cuda.sh
# 또는
chmod +x activate_cuda.sh
./activate_cuda.sh

# 또는 수동으로 설정
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### 주의 사항

```
1. 아래의 위치에 데이터셋을 첨부해야 함
첨부 방식과 다운로드 링크는 ROBOSAC의 README.txt를 참고
ROBOSAC\coperception\coperception\datasets\

2. 아래의 위치에 모델 체크포인트를 첨부해야 함
첨부 방식과 다운로드 링크는 ROBOSAC의 README.txt를 참고
ROBOSAC\coperception\ckpt\meanfusion\
```
