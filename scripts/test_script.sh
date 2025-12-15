export CUDA_VISIBLE_DEVICES=4

### Data description
seq_len=96        # 96, 192, 336, 720
feature_dim=1

### Training description
learning_rate=0.0001
#learning_rate=0.01
batch_size=32
train_epochs=100

### Model description
model_name=Ours
interval=0.1
hidden_dim=32
num_heads=4
num_dit_block=4

### 🔥 추가 옵션들 — 여기를 바꾸면 바로 적용됨 🔥
use_ma_start=0          # 0 = 기존 x 확장/ 1 = peaking(EMA) / 2 = predictor 사용

lambda_mu=0.0           # μ predictor loss weight / 사용시 1.0
lambda_traj=1.0
lambda_end=1.0

variate=S

fig_tag="12_11_newmethod"   # ★ 추가: 원하는 이름

python -u run.py \
  --task_name test \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96_$variate \
  --model $model_name \
  --train_epochs $train_epochs \
  --data ETTm2 \
  --features $variate \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --feature_dim $feature_dim \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --interval $interval \
  --hidden_dim $hidden_dim \
  --num_heads $num_heads \
  --num_dit_block $num_dit_block \
  --fig_tag $fig_tag \
    \
  --use_ma_start $use_ma_start \
  --lambda_mu $lambda_mu \
  --lambda_traj $lambda_traj \
  --lambda_end $lambda_end
