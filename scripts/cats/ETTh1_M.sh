#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

### Training description
learning_rate=0.0001
batch_size=32
train_epochs=100
patience=50

### Model description - Using CATS version
model_name=Ours_CATS
interval=0.1
hidden_dim=128
num_heads=8
num_dit_block=4

### CATS specific
num_aux=8  # Number of auxiliary time series

### Loss configuration
use_ma_start=0
lambda_mu=0.0
lambda_traj=1.0
lambda_end=1.0

### Multivariate settings
variate=M
feature_dim=7

fig_tag="cats"
exp_tag="ETTh1_M"

# Array of prediction lengths to test
pred_lengths=(96 192 336 720)

echo "================================"
echo "Starting ETTh1 (Multivariate) with CATS"
echo "================================"
for pred_len in "${pred_lengths[@]}"; do
    echo "Running ETTh1 with pred_len=$pred_len"
    python -u run.py \
      --task_name test \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_${pred_len}_${variate} \
      --model $model_name \
      --train_epochs $train_epochs \
      --data ETTh1 \
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
      --num_aux $num_aux \
      --fig_tag $fig_tag \
      --exp_tag "$exp_tag" \
      --use_ma_start $use_ma_start \
      --lambda_mu $lambda_mu \
      --lambda_traj $lambda_traj \
      --lambda_end $lambda_end \
      --patience $patience
done

echo "================================"
echo "ETTh1 (Multivariate) CATS experiments completed!"
echo "================================"
