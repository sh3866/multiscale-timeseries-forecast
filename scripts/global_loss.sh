export CUDA_VISIBLE_DEVICES=4

### Data description
seq_len=96        # 96, 192, 336, 720
feature_dim=7

### Training description
learning_rate=0.001
#learning_rate=0.01
batch_size=64
train_epochs=50

### Model description
model_name=Ours
interval=0.02
hidden_dim=32
num_heads=4
num_dit_block=4

python -u run.py \
  --task_name global_loss \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'96 \
  --model $model_name \
  --train_epochs $train_epochs \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --feature_dim $feature_dim \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --interval $interval \
  --hidden_dim $hidden_dim \
  --num_heads $num_heads \
  --num_dit_block $num_dit_block
