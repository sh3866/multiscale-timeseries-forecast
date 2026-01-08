"""
Simplified WandB Sweep Agent
데이터셋 설정을 자동으로 처리하여 YAML 파일 중복 제거

Usage:
    # 1. Create sweep (한 번만 실행)
    wandb sweep sweeps/sweep_config.yaml

    # 2. Run agent
    python sweeps/sweep_agent.py \
        --sweep_id <SWEEP_ID> \
        --dataset ETTm2 \
        --variate S \
        --count 50 \
        --gpu 0
"""

import argparse
import os
import sys
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.test import Test


# 데이터셋 설정 매핑 (YAML 중복 제거!)
DATASET_CONFIG = {
    'ETTm1': {
        'S': {'root_path': './dataset_matsd/ETTm1/', 'data_path': 'ETTm1.csv', 'data': 'ETTm1', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/ETTm1/', 'data_path': 'ETTm1.csv', 'data': 'ETTm1', 'feature_dim': 7},
    },
    'ETTm2': {
        'S': {'root_path': './dataset_matsd/ETTm2/', 'data_path': 'ETTm2.csv', 'data': 'ETTm2', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/ETTm2/', 'data_path': 'ETTm2.csv', 'data': 'ETTm2', 'feature_dim': 7},
    },
    'ETTh1': {
        'S': {'root_path': './dataset_matsd/ETTh1/', 'data_path': 'ETTh1.csv', 'data': 'ETTh1', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/ETTh1/', 'data_path': 'ETTh1.csv', 'data': 'ETTh1', 'feature_dim': 7},
    },
    'ETTh2': {
        'S': {'root_path': './dataset_matsd/ETTh2/', 'data_path': 'ETTh2.csv', 'data': 'ETTh2', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/ETTh2/', 'data_path': 'ETTh2.csv', 'data': 'ETTh2', 'feature_dim': 7},
    },
    'weather': {
        'S': {'root_path': './dataset_matsd/weather/', 'data_path': 'weather.csv', 'data': 'custom', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/weather/', 'data_path': 'weather.csv', 'data': 'custom', 'feature_dim': 21},
    },
    'electricity': {
        'S': {'root_path': './dataset_matsd/electricity/', 'data_path': 'electricity.csv', 'data': 'custom', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/electricity/', 'data_path': 'electricity.csv', 'data': 'custom', 'feature_dim': 321},
    },
    'traffic': {
        'S': {'root_path': './dataset_matsd/traffic/', 'data_path': 'traffic.csv', 'data': 'custom', 'feature_dim': 1},
        'M': {'root_path': './dataset_matsd/traffic/', 'data_path': 'traffic.csv', 'data': 'custom', 'feature_dim': 862},
    },
}


def train():
    """
    WandB Sweep에서 호출되는 학습 함수
    """
    with wandb.init() as run:
        # Get parameters from sweep config (YAML)
        config = wandb.config

        # Get dataset config from global variables
        dataset = DATASET_NAME
        variate = VARIATE

        if dataset not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIG.keys())}")
        if variate not in DATASET_CONFIG[dataset]:
            raise ValueError(f"Unknown variate: {variate}. Available: {list(DATASET_CONFIG[dataset].keys())}")

        dataset_cfg = DATASET_CONFIG[dataset][variate]

        # Create args object
        class Args:
            pass

        args = Args()

        # Set parameters from sweep config
        for key, value in config.items():
            setattr(args, key, value)

        # Override with dataset-specific config
        args.root_path = dataset_cfg['root_path']
        args.data_path = dataset_cfg['data_path']
        args.data = dataset_cfg['data']
        args.feature_dim = dataset_cfg['feature_dim']
        args.features = variate

        # Channel independent: 0 for univariate (S), 1 for multivariate (M)
        args.channel_independent = 1 if variate == 'M' else 0

        # GPU settings
        args.use_gpu = True
        args.gpu = GPU_NUMBER
        args.use_multi_gpu = False
        args.devices = '0,1,2,3'

        # Build model_id
        dataset_name = args.data if args.data != 'custom' else args.data_path.replace('.csv', '')
        args.model_id = f'{dataset_name}_{args.seq_len}_{args.pred_len}_{args.features}'

        # Update wandb run name
        run_name = f"{dataset_name}_{args.seq_len}_{args.pred_len}_{args.features}_sweep"
        run.name = run_name

        # Print config
        print(f'\n{"="*80}')
        print(f'Running Sweep: {run.name}')
        print(f'Dataset: {dataset} ({variate})')
        print(f'Sweep Params:')
        sweep_params = {k: v for k, v in config.items() if k in [
            'hidden_dim', 'num_heads', 'num_dit_block', 'interval',
            'learning_rate', 'batch_size', 'lambda_traj', 'lambda_end'
        ]}
        for k, v in sweep_params.items():
            print(f'  {k}: {v}')
        print(f'{"="*80}\n')

        # Create experiment instance
        exp = Test(args)

        # Setting name (wandb run.id 추가로 각 run마다 고유한 checkpoint 경로)
        setting = f'{dataset_name}_{args.seq_len}_{args.pred_len}_{args.features}/{run.id}'

        # Train
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

        # Test
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        test_metrics = exp.test(setting, test=1)

        # Log final test metrics
        if test_metrics:
            wandb.log({
                'test/mse': test_metrics.get('mse', 0),
                'test/mae': test_metrics.get('mae', 0),
            })

        # Clean up GPU memory
        import torch
        import gc

        del exp
        gc.collect()
        torch.cuda.empty_cache()

        print(f'Memory cleaned up after run: {run.name}')


def main():
    parser = argparse.ArgumentParser(description='Simplified WandB Sweep Agent')
    parser.add_argument('--sweep_id', type=str, required=True,
                        help='WandB sweep ID (format: entity/project/sweep_id)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIG.keys()),
                        help='Dataset name')
    parser.add_argument('--variate', type=str, required=True,
                        choices=['S', 'M'],
                        help='S (univariate) or M (multivariate)')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of sweep runs')
    parser.add_argument('--project', type=str, default='time_series_forecasting',
                        help='WandB project name')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number to use')

    args = parser.parse_args()

    # Store as global variables so train() can access them
    global DATASET_NAME, VARIATE, GPU_NUMBER
    DATASET_NAME = args.dataset
    VARIATE = args.variate
    GPU_NUMBER = args.gpu

    print(f"\n{'='*80}")
    print(f"Starting WandB Sweep Agent")
    print(f"{'='*80}")
    print(f"Sweep ID: {args.sweep_id}")
    print(f"Dataset: {args.dataset} ({args.variate})")
    print(f"Project: {args.project}")
    print(f"GPU: {args.gpu}")
    print(f"Number of runs: {args.count}")
    print(f"{'='*80}\n")

    # Start sweep agent
    wandb.agent(
        args.sweep_id,
        function=train,
        count=args.count,
        project=args.project
    )


if __name__ == '__main__':
    main()
