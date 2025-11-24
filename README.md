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

### GitHub에 업로드하기

```bash
# 1. WSL에서 Git 설정 (처음 한 번만)
git config --global --add safe.directory /mnt/c/Users/dnrua/Desktop/EgoLoc
git config --global core.fileMode false
git config --global core.autocrlf input

# 2. 사용자 정보 설정 (처음 한 번만)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 3. Git 저장소 초기화 (이미 초기화되어 있으면 생략)
git init

# 4. 파일 추가
git add .

# 5. 첫 커밋
git commit -m "Initial commit: EgoLoc project"

# 6. GitHub에서 새 저장소 생성 후 (웹에서)
# 7. 원격 저장소 추가 및 푸시
git remote add origin https://github.com/yourusername/EgoLoc.git
git branch -M main
git push -u origin main
```

**참고**: 
- `.gitignore` 파일이 이미 있어서 불필요한 파일은 자동으로 제외됩니다
- 대용량 파일(데이터셋, 체크포인트)은 GitHub에 직접 업로드하지 마세요
- GitHub에서 새 저장소를 먼저 생성한 후 원격 저장소 URL을 사용하세요

### 주의 사항

```
1. 아래의 위치에 데이터셋을 첨부해야 함
첨부 방식과 다운로드 링크는 ROBOSAC의 README.txt를 참고
ROBOSAC\coperception\coperception\datasets\

2. 아래의 위치에 모델 체크포인트를 첨부해야 함
첨부 방식과 다운로드 링크는 ROBOSAC의 README.txt를 참고
ROBOSAC\coperception\ckpt\meanfusion\
```
