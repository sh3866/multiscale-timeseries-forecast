import argparse
import random

import numpy as np
import torch
import wandb

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.global_loss import Global_Loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def get_args():
    parser = argparse.ArgumentParser(description='TimeMixer')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--seed', type=int, default=0, help='Seed for training')

    # data loader
    parser.add_argument('--data', type=str, required=True, help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, ms:milliseconds], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--feature_dim', type=int, default=7, help='input feature dimension size')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--interval', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_dit_block', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.125, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--drop_last', type=bool, default=True, help='drop last')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--comment', type=str, default='none', help='com')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    set_seed(args.seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    wandb.init(project='time_series_diffusion')
    wandb.config.update(args)

    print('Args in experiment:')
    print(args)
    
    tasks = {
        'long_term_forecast': Exp_Long_Term_Forecast,
        'global_loss': Global_Loss
    }
    exp = tasks[args.task_name](args)

    setting = '{}_{}_{}_{}_sl{}'.format(
        args.task_name,
        args.model_id,
        args.comment,
        args.model,
        args.data,
        args.seq_len)

    if args.is_training:
        print('------------- start training : {} -------------'.format(setting))
        exp.train(setting)
    
    print('------------- testing : {} -------------'.format(setting))
    exp.test(setting, test=True)
    torch.cuda.empty_cache()
