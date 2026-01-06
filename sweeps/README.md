# Simplified WandB Sweep Guide

기존의 복잡한 구조를 간소화했습니다. 이제 **단 3개 파일**로 모든 sweep을 관리할 수 있습니다!

## 📁 파일 구조 (간소화됨!)

```
sweeps/
├── sweep_config.yaml      # 단 1개의 YAML 설정 (모든 데이터셋 공통)
├── sweep_agent.py         # 데이터셋 자동 설정 agent
├── run_sweeps.sh          # 병렬 실행 스크립트
└── README.md              # 이 파일
```

**기존**: 12개 YAML + 복잡한 스크립트들
**현재**: 3개 파일만!

## 🚀 Quick Start

### 1. Sweep 생성 (첫 실행 시 한 번만)

```bash
cd /data_seoul/sunghyun/time_series_forecasting
wandb sweep sweeps/sweep_config.yaml
```

출력 예시:
```
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/username/time_series_forecasting/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent username/time_series_forecasting/abc123xyz
```

**중요**: `username/time_series_forecasting/abc123xyz` 형태의 전체 ID를 복사하세요!

### 2. 병렬 실행 스크립트 설정

`sweeps/run_sweeps.sh`를 열어서 sweep ID를 설정:

```bash
# Line 28 수정
SWEEP_ID="sunghyunchoi-postech/time_series_forecasting/abc123xyz"
```

### 3. 실행!

```bash
bash sweeps/run_sweeps.sh
```

그게 끝입니다! 🎉

## 💡 개별 데이터셋 실행

전체가 아닌 특정 데이터셋만 실행하려면:

```bash
# ETTm2 univariate만
python sweeps/sweep_agent.py \
    --sweep_id sunghyunchoi-postech/time_series_forecasting/abc123 \
    --dataset ETTm2 \
    --variate S \
    --count 50 \
    --gpu 0

# Weather multivariate만
python sweeps/sweep_agent.py \
    --sweep_id sunghyunchoi-postech/time_series_forecasting/abc123 \
    --dataset weather \
    --variate M \
    --count 50 \
    --gpu 1
```

## 📊 지원 데이터셋

모든 데이터셋이 `sweep_agent.py`에 자동 설정되어 있습니다:

| Dataset | Univariate (S) | Multivariate (M) | M Feature Dim |
|---------|----------------|------------------|---------------|
| ETTm1   | ✅ | ✅ | 7 |
| ETTm2   | ✅ | ✅ | 7 |
| ETTh1   | ✅ | ✅ | 7 |
| ETTh2   | ✅ | ✅ | 7 |
| weather | ✅ | ✅ | 21 |
| electricity | ✅ | ✅ | 321 |
| traffic | ✅ | ✅ | 862 |

## 🎯 Sweep 파라미터

`sweep_config.yaml`에 정의됨 (모든 데이터셋 공통):

### Sweep할 하이퍼파라미터
- `hidden_dim`: [32, 64, 128]
- `num_heads`: [4, 8, 16]
- `num_dit_block`: [2, 4, 6, 8]
- `interval`: [0.01, 0.05, 0.1]
- `learning_rate`: [0.00005, 0.0001]
- `batch_size`: [32, 64]
- `lambda_traj`: [1.0]
- `lambda_end`: [1.0]

### 고정 파라미터
- `seq_len`: 96
- `pred_len`: 96
- `train_epochs`: 100
- `patience`: 50
- 기타 모델/학습 설정

## 🔧 파라미터 수정

### Sweep 범위 변경

`sweeps/sweep_config.yaml` 수정:

```yaml
parameters:
  hidden_dim:
    values: [32, 64, 128, 256]  # 256 추가

  learning_rate:
    values: [0.00001, 0.00005, 0.0001, 0.0002]  # 범위 확장
```

### 새 데이터셋 추가

`sweeps/sweep_agent.py`의 `DATASET_CONFIG` 딕셔너리에 추가:

```python
DATASET_CONFIG = {
    # ... existing datasets ...
    'new_dataset': {
        'S': {'root_path': './dataset_matsd/new/', 'data_path': 'new.csv', 'data': 'custom', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/new/', 'data_path': 'new.csv', 'data': 'custom', 'feature_dim': 100},
    },
}
```

## 📈 결과 확인

WandB 대시보드에서:
- 실시간 학습 진행 상황
- 하이퍼파라미터별 성능 비교
- Parallel coordinates plot
- Best run 자동 추천

## 🛠️ 문제 해결

### Q: "YOUR_SWEEP_ID_HERE" 에러
**A**: `run_sweeps.sh`에서 SWEEP_ID를 실제 ID로 교체하세요

### Q: 특정 GPU만 사용하고 싶음
**A**: `run_sweeps.sh`에서 필요한 부분만 주석 해제하거나, 직접 `sweep_agent.py` 실행

### Q: Sweep 파라미터 변경했는데 반영 안됨
**A**: 새로운 sweep을 생성해야 합니다 (`wandb sweep ...` 다시 실행)

## 📚 기존 구조와 비교

### Before (복잡함)
```
sweeps/
├── single/
│   ├── ETTm2_S.yaml      (116줄)
│   ├── ETTh2_S.yaml      (116줄)
│   ├── weather_S.yaml    (116줄)
│   ├── electricity_S.yaml (116줄)
│   └── traffic_S.yaml    (116줄)
├── multi/
│   ├── ETTm1_M.yaml      (116줄)
│   ├── ETTm2_M.yaml      (116줄)
│   ├── ETTh1_M.yaml      (116줄)
│   ├── ETTh2_M.yaml      (116줄)
│   ├── weather_M.yaml    (116줄)
│   ├── electricity_M.yaml (116줄)
│   └── traffic_M.yaml    (116줄)
├── sweep_agent_dataset.py
├── create_all_sweeps.sh
├── run_all_single.sh
└── run_all_multi.sh
```
**총**: 12개 YAML + 4개 스크립트 = **16개 파일**
**문제**: 90% 중복 내용, sweep ID 수동 관리

### After (간소화됨!)
```
sweeps/
├── sweep_config.yaml     (1개 YAML, 모든 데이터셋 공통)
├── sweep_agent.py        (데이터셋 자동 설정)
├── run_sweeps.sh         (병렬 실행)
└── README.md
```
**총**: **3개 핵심 파일**
**장점**: 중복 제거, 유지보수 간편, sweep ID 한 곳에서 관리

## ⚡ 핵심 개선 사항

1. **YAML 파일 12개 → 1개**: 모든 데이터셋이 동일한 sweep 설정 사용
2. **데이터셋 설정 자동화**: Python 딕셔너리로 자동 매핑
3. **Sweep ID 한 곳에서 관리**: `run_sweeps.sh` 하나만 수정
4. **더 간단한 실행**: 파일 찾아다니지 않아도 됨

Happy sweeping! 🚀
