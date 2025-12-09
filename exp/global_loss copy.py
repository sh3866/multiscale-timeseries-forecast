import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models import get_model

warnings.filterwarnings('ignore')


class Global_Loss(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.alphas = torch.arange(0.0, 1.0 + self.args.interval, self.args.interval)
        
        # # 새 버전: 비선형 스케줄
        # num_alphas = int(1 / self.args.interval) + 1
        # gamma = 2.0   # 초반부 촘촘하게
        # alphas_np = np.linspace(0, 1, num_alphas) ** (1.0 / gamma)
        # self.alphas = torch.tensor(alphas_np, dtype=torch.float32)
        
        # # 🔹 디버깅 출력
        # print("[DEBUG] self.alphas shape:", self.alphas.shape)
        # print("[DEBUG] self.alphas values:\n", self.alphas.numpy())

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if self.args.use_multi_gpu:
                print('Use GPU: cuda{}'.format(self.args.device_ids))
            else:
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = get_model(self.args.model, self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def compute_ema_sequences(self, x, interval):
        batch_size, seq_len, feature_dim = x.shape
        
        ema_outputs = torch.zeros((batch_size, len(self.alphas), seq_len, feature_dim), dtype=x.dtype, device=x.device)

        for i, alpha in enumerate(self.alphas):
            ema = x[:, 0, :]  # (batch, feature_dim), init with first x_t
            ema_seq = [ema.unsqueeze(1)]  # time t=0

            for t in range(1, seq_len):
                ema = alpha * ema + (1 - alpha) * x[:, t, :]
                ema_seq.append(ema.unsqueeze(1))

            ema_seq = torch.cat(ema_seq, dim=1)  # (batch, seq_len, feature_dim)
            ema_outputs[:, i, :, :] = ema_seq
        
        alpha_values = self.alphas.unsqueeze(0).expand(batch_size, -1)
        
        return ema_outputs, alpha_values

    def process_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        _, seq_len, feature_dim = batch_x.shape
        T_pred = batch_y.shape[1]
        
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.float(), batch_y.float(), batch_x_mark.float(), batch_y_mark.float()

        if 'PEMS' == self.args.data or 'Solar' == self.args.data:
            batch_x_mark = None
            batch_y_mark = None
        
        batch_ema_y, alpha_values = self.compute_ema_sequences(torch.cat([batch_x[:, -1].unsqueeze(1), batch_y], dim=1), interval=self.args.interval)
        batch_ema_y = batch_ema_y[:, :, 1:]
        
        batch_x = batch_x.unsqueeze(1).expand(-1, int(1 / self.args.interval), -1, -1).contiguous().view(-1, seq_len, feature_dim).to(self.device)
        batch_ema_y_prev = batch_ema_y[:, 1:].contiguous().view(-1, T_pred, feature_dim).to(self.device)
        batch_ema_y_target = batch_ema_y[:, :-1].contiguous().view(-1, T_pred, feature_dim).to(self.device)
        alpha_values = alpha_values[:, 1:].contiguous().view(-1).to(self.device)
        
        return batch_x, batch_x_mark, batch_y_mark, batch_ema_y_prev, batch_ema_y_target, alpha_values

    def process_batch_for_test(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.float(), batch_y.float(), batch_x_mark.float(), batch_y_mark.float()

        if 'PEMS' == self.args.data or 'Solar' == self.args.data:
            batch_x_mark = None
            batch_y_mark = None
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    def sampling(self, x, x_mark, y_mark):
        batch_size = x.shape[0]
        
        x = x.to(self.device)
        x_mark = x_mark.to(self.device) if x_mark is not None else None
        y_mark = y_mark.to(self.device) if y_mark is not None else None
        
        output_t = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1)

        for alpha in torch.flip(self.alphas[1:], dims=[0]):
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))
        
        return output_t
    
    # 중간 alpha부터 복원
    def sampling_from_alpha(self, x, x_mark, y_mark, start_alpha_idx):
        batch_size = x.shape[0]
        
        x = x.to(self.device)
        x_mark = x_mark.to(self.device) if x_mark is not None else None
        y_mark = y_mark.to(self.device) if y_mark is not None else None

        # 초기 상태는 "해당 alpha 시점의 corrupted y"
        output_t = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1)

        # α 리스트에서 start_alpha_idx 부터만 복원
        alphas = torch.flip(self.alphas[1:], dims=[0])
        alphas = alphas[start_alpha_idx:]  

        for alpha in alphas:
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))

        return output_t

    
    # differentiable sampling that returns all intermediate predictions as a tensor
    def sampling_with_intermediates_tensor(self, x, x_mark, y_mark):
        batch_size = x.shape[0]
        
        x = x.to(self.device)
        x_mark = x_mark.to(self.device) if x_mark is not None else None
        y_mark = y_mark.to(self.device) if y_mark is not None else None

        output_t = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1)
        preds = []  # list of tensors, one per alpha step (reverse order)

        for alpha in torch.flip(self.alphas[1:], dims=[0]):
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))
            preds.append(output_t)

        # stack along step dimension: (B, num_steps=A-1, T_pred, C)
        return torch.stack(preds, dim=1)
    
    # sampling for training
    def sampling_with_intermediates(self, x, x_mark, y_mark):
        batch_size = x.shape[0]
        x = x.to(self.device)
        x_mark = x_mark.to(self.device) if x_mark is not None else None
        y_mark = y_mark.to(self.device) if y_mark is not None else None

        output_t = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1)
        preds = []  # α step별 결과 기록

        for alpha in torch.flip(self.alphas[1:], dims=[0]):
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))
            preds.append(output_t.detach().cpu().numpy())

        return preds

    
    # 이미지 저장 폴더 이름 설정
    def _run_name(self, fallback: str) -> str:
        try:
            if wandb.run is not None:
                return (wandb.run.name or wandb.run.id)
        except Exception:
            pass
        return fallback

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.process_batch_for_test(batch_x, batch_y, batch_x_mark, batch_y_mark)
                
                pred_y = self.sampling(batch_x, batch_x_mark, batch_y_mark)
                true_y = batch_y.to(self.device)

                # 로스 계산
                loss = criterion(pred_y, true_y)
                total_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 이미지 저장 경로 세팅
        run_name = self._run_name(setting)
        train_fig_dir = os.path.join('./figs', run_name, 'train')
        os.makedirs(train_fig_dir, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                model_optim.zero_grad()

                # === Step 1. 배치 준비 ===
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # === Step 2. 랜덤 α 샘플링 ===
                rand_idx = torch.randint(0, len(self.alphas), (1,), device=self.device).item()
                alpha = self.alphas[rand_idx]

                # === Step 3. forward EMA 적용 ===
                #  y_alpha = EMA_alpha(x0 + future)
                ema = batch_x[:, -1, :]   # 초기값: 과거 마지막 값
                ema_seq = [ema.unsqueeze(1)]
                for t in range(batch_y.shape[1]):  # pred_len 만큼
                    ema = alpha * ema + (1 - alpha) * batch_y[:, t, :]
                    ema_seq.append(ema.unsqueeze(1))
                y_alpha = torch.cat(ema_seq[1:], dim=1)  # (B, pred_len, C)

                # === Step 4. 모델 복원 ===
                pred_y = self.model(y_alpha, batch_x, alpha.expand(batch_x.size(0)))

                # === Step 5. Loss 계산 (원본과 비교) ===
                loss = criterion(pred_y, batch_y)

                # === Step 6. Backward ===
                loss.backward()
                model_optim.step()

                pbar.set_postfix(loss=loss.item())

                # === WandB 로그 ===
                wandb.log({
                    "epoch": epoch,
                    "iteration": i,
                    "train/loss": loss.item()
                })

            # === Validation ===
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.test(setting, test=False, epoch=epoch)

            wandb.log({
                "epoch": epoch,
                "val/loss": vali_loss,
                "test/loss": test_loss,
            })

            print(f"Epoch: {epoch+1} | Vali Loss: {vali_loss:.6f} Test Loss: {test_loss:.6f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test, epoch=None):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        run_name = self._run_name(setting)
        test_fig_dr = os.path.join('./figs', run_name, 'test')
        os.makedirs(test_fig_dr, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.process_batch_for_test(batch_x, batch_y, batch_x_mark, batch_y_mark)
                
                outputs = self.sampling(batch_x, batch_x_mark, batch_y_mark)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                
                # === test 결과 시각화 코드 ===
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    
                    # gt0 = np.concatenate((input[0, :, 0], true[0, :, 0]), axis=0)
                    # pd0 = np.concatenate((input[0, :, 0], pred[0, :, 0]), axis=0)
                    
                    # gt1 = np.concatenate((input[0, :, 1], true[0, :, 1]), axis=0)
                    # pd1 = np.concatenate((input[0, :, 1], pred[0, :, 1]), axis=0)
                    
                    # gt2 = np.concatenate((input[0, :, 2], true[0, :, 2]), axis=0)
                    # pd2 = np.concatenate((input[0, :, 2], pred[0, :, 2]), axis=0)
                    
                    # gt3 = np.concatenate((input[0, :, 3], true[0, :, 3]), axis=0)
                    # pd3 = np.concatenate((input[0, :, 3], pred[0, :, 3]), axis=0)
                    
                    # gt4 = np.concatenate((input[0, :, 4], true[0, :, 4]), axis=0)
                    # pd4 = np.concatenate((input[0, :, 4], pred[0, :, 4]), axis=0)
                    
                    # gt5 = np.concatenate((input[0, :, 5], true[0, :, 5]), axis=0)
                    # pd5 = np.concatenate((input[0, :, 5], pred[0, :, 5]), axis=0)
                    
                    if epoch is not None:
                        visual(gt, pd, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}.pdf"))
                        
                        # visual(gt0, pd0, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}_0.pdf"))
                        # visual(gt1, pd1, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}_1.pdf"))
                        # visual(gt2, pd2, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}_2.pdf"))
                        # visual(gt3, pd3, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}_3.pdf"))
                        # visual(gt4, pd4, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}_4.pdf"))
                        # visual(gt5, pd5, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}_5.pdf"))
                    else:
                        visual(gt, pd, os.path.join(test_fig_dr, f"{str(i)}.pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # mae, mse, rmse, mape, mspe = metric(preds[..., -1:], trues[..., -1:])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        wandb.log({
            'test/mse': mse,
            'test/mae': mae,
        })
        return
