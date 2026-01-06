#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

### Training description
learning_rate=0.0001
batch_size=8
train_epochs=100
patience=50

### Model description - Using Channel Mixing version
model_name=Ours_CM
interval=0.1
hidden_dim=128
num_heads=8
num_dit_block=4

### Loss configuration
use_ma_start=0
lambda_mu=0.0
lambda_traj=1.0
lambda_end=1.0

### Multivariate settings
variate=M
feature_dim=862

fig_tag="channel_mixing"
exp_tag="traffic_M"

# Array of prediction lengths to test
pred_lengths=(96 192 336 720)

echo "================================"
echo "Starting Traffic (Multivariate) with Channel Mixing"
echo "================================"
for pred_len in "${pred_lengths[@]}"; do
    echo "Running Traffic with pred_len=$pred_len"
    python -u run.py \
      --task_name test \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_96_${pred_len}_${variate} \
      --model $model_name \
      --train_epochs $train_epochs \
      --data custom \
      --features $variate \
      --seq_len 96 \
      --label_len 0 \
      --pred_len $pred_len \
      --feature_dim $feature_dim \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --interval $interval \
      --hidden_dim $hidden_dim \
      --num_heads $num_heads \
      --num_dit_block $num_dit_block \
      --fig_tag $fig_tag \
      --exp_tag "$exp_tag" \
      --use_ma_start $use_ma_start \
      --lambda_mu $lambda_mu \
      --lambda_traj $lambda_traj \
      --lambda_end $lambda_end \
      --patience $patience
done

echo "================================"
echo "Traffic (Multivariate) Channel Mixing experiments completed!"
echo "================================"
