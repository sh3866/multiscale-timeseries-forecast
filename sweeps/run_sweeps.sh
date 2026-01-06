#!/bin/bash

# Simplified Sweep Runner
# 하나의 sweep ID로 여러 데이터셋을 병렬 실행

# Activate conda environment
source /data_seoul/sunghyun/anaconda3/etc/profile.d/conda.sh
conda activate timemixer

cd /data_seoul/sunghyun/time_series_forecasting

# Create log directory
mkdir -p logs/sweeps

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Simplified WandB Sweep Runner                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Step 1: Create sweep (첫 실행 시만 필요)"
echo "  wandb sweep sweeps/sweep_config.yaml"
echo ""
echo "Step 2: Copy the sweep ID from the output"
echo "  Example: sunghyunchoi-postech/time_series_forecasting/abc123"
echo ""
echo "Step 3: Edit this script and set SWEEP_ID below"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ========== EDIT HERE: Set your sweep ID ==========
SWEEP_ID="sunghyunchoi-postech/time_series_forecasting/cvt7jqlu"
# Example: SWEEP_ID="sunghyunchoi-postech/time_series_forecasting/abc123"

if [ "$SWEEP_ID" == "YOUR_SWEEP_ID_HERE" ]; then
    echo "❌ Error: Please set SWEEP_ID in this script!"
    echo ""
    echo "Instructions:"
    echo "1. Run: wandb sweep sweeps/sweep_config.yaml"
    echo "2. Copy the sweep ID from output"
    echo "3. Edit this script and replace YOUR_SWEEP_ID_HERE"
    echo "4. Run this script again"
    exit 1
fi

# ========== Configuration ==========
COUNT=50  # Number of runs per dataset

echo "Sweep ID: $SWEEP_ID"
echo "Runs per dataset: $COUNT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting Sweeps..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# SINGLE (Univariate) datasets
echo "→ [GPU 0] ETTm2_S"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset ETTm2 \
    --variate S \
    --count $COUNT \
    --gpu 0 \
    > logs/sweeps/ETTm2_S.log 2>&1 &
PID1=$!

echo "→ [GPU 1] ETTh2_S"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset ETTh2 \
    --variate S \
    --count $COUNT \
    --gpu 1 \
    > logs/sweeps/ETTh2_S.log 2>&1 &
PID2=$!

echo "→ [GPU 2] weather_S"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset weather \
    --variate S \
    --count $COUNT \
    --gpu 2 \
    > logs/sweeps/weather_S.log 2>&1 &
PID3=$!

echo "→ [GPU 3] electricity_S"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset electricity \
    --variate S \
    --count $COUNT \
    --gpu 3 \
    > logs/sweeps/electricity_S.log 2>&1 &
PID4=$!

echo "→ [GPU 4] traffic_S"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset traffic \
    --variate S \
    --count $COUNT \
    --gpu 4 \
    > logs/sweeps/traffic_S.log 2>&1 &
PID5=$!

# MULTI (Multivariate) datasets
echo "→ [GPU 5] ETTm2_M"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset ETTm2 \
    --variate M \
    --count $COUNT \
    --gpu 5 \
    > logs/sweeps/ETTm2_M.log 2>&1 &
PID6=$!

echo "→ [GPU 6] weather_M"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset weather \
    --variate M \
    --count $COUNT \
    --gpu 6 \
    > logs/sweeps/weather_M.log 2>&1 &
PID7=$!

echo "→ [GPU 7] electricity_M"
nohup python sweeps/sweep_agent.py \
    --sweep_id $SWEEP_ID \
    --dataset electricity \
    --variate M \
    --count $COUNT \
    --gpu 7 \
    > logs/sweeps/electricity_M.log 2>&1 &
PID8=$!

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              All Sweeps Started!                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "PIDs: $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Monitoring Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "View logs:"
echo "  tail -f logs/sweeps/ETTm2_S.log"
echo "  tail -f logs/sweeps/weather_M.log"
echo ""
echo "Check processes:"
echo "  ps aux | grep sweep_agent.py"
echo ""
echo "Check GPU usage:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "Stop all sweeps:"
echo "  kill $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8"
echo "  pkill -f sweep_agent.py"
echo ""
echo "WandB Dashboard:"
echo "  https://wandb.ai/YOUR_ENTITY/time_series_forecasting/sweeps"
echo ""
