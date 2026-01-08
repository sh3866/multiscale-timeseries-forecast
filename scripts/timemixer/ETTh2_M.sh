#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

### Training description
learning_rate=0.01
batch_size=16
train_epochs=100
patience=10

### Model description (TimeMixer specific)
model_name=TimeMixer_MA
seq_len=96
e_layers=2
d_model=16
d_ff=32
down_sampling_layers=3
down_sampling_window=2
down_sampling_method=avg

### MA-Diffusion specific
interval=0.1

### Loss configuration
use_ma_start=0
lambda_mu=0.0
lambda_traj=1.0
lambda_end=0.0

### Channel mode: 0=channel-mixing, 1=channel-independent
channel_independent=1

variate=M
enc_in=7
c_out=7

fig_tag="01_09_timemixer"
exp_tag="second"

# Array of prediction lengths to test
pred_lengths=(96)

echo "================================"
echo "Starting ETTh2 (Multivariate) TimeMixer_MA experiments"
echo "================================"
for pred_len in "${pred_lengths[@]}"; do
    echo "Running ETTh2 with pred_len=$pred_len"
    python -u run.py \
      --task_name test \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_${seq_len}_${pred_len}_${variate} \
      --model $model_name \
      --train_epochs $train_epochs \
      --data ETTh2 \
      --features $variate \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in $enc_in \
      --c_out $c_out \
      --d_model $d_model \
      --d_ff $d_ff \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_window $down_sampling_window \
      --down_sampling_method $down_sampling_method \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --interval $interval \
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
echo "ETTh2 (Multivariate) TimeMixer_MA experiments completed!"
echo "================================"
