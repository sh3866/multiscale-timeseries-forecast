#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

### Training description
learning_rate=0.0001
batch_size=32
train_epochs=100
patience=100

### Model description
model_name=TimeMixer_MA
interval=0.1
hidden_dim=128
num_heads=8
num_dit_block=4

### Loss configuration
use_ma_start=0
lambda_mu=0.0
lambda_traj=1.0
lambda_end=0.0

### Channel mode: 0=channel-mixing, 1=channel-independent
channel_independent=1

variate=M
feature_dim=7

fig_tag="01_09_timemixer"
exp_tag="first"  # 추가 태그 (예: parameter_test, ablation 등) - 비워두면 기본값

# Array of prediction lengths to test
seq_lengths=(96)

echo "================================"
echo "Starting ETTh2 (Multivariate) experiments"
echo "================================"
for pred_len in "${seq_lengths[@]}"; do
    echo "Running ETTh2 with pred_len=$pred_len"
    python -u run.py \
      --task_name test \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_${pred_len}_${variate} \
      --model $model_name \
      --train_epochs $train_epochs \
      --data ETTh2 \
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
      --patience $patience \
      --channel_independent $channel_independent
done

echo "================================"
echo "ETTh2 (Multivariate) experiments completed!"
echo "================================"
