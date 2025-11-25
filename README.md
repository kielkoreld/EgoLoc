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

#로그 파일 생성
... --log 추가
```

### 환경 설치

```bash
# 자동 설치 (권장)
chmod +x setup_egoloc.sh
./setup_egoloc.sh

# 설치 완료 후 환경 활성화
conda activate EgoLoc
```

### CUDA 라이브러리 경로 설정 (WSL 환경)

```bash
# 1. 자동 설정 스크립트 실행 (한 번만 실행하면 됩니다)
chmod +x setup_cuda_env.sh
./setup_cuda_env.sh

# 2. 이후 환경 활성화 시마다 자동으로 설정됩니다
conda activate EgoLoc
```

**수동 설정이 필요한 경우** (자동 설정이 작동하지 않을 때만):

```bash
conda activate EgoLoc
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
