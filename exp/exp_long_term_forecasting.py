import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models import get_model

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.alphas = torch.arange(0.0, 1.0 + self.args.interval, self.args.interval, device=self.device)


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

    # def _select_criterion(self):
    #     if self.args.data == 'PEMS':
    #         criterion = nn.L1Loss()
    #     else:
    #         criterion = nn.MSELoss()
    #     return criterion
    
    def _select_criterion(self):
        # args.loss가 있으면 반영 (기본 MSE)
        if hasattr(self.args, "loss") and str(self.args.loss).upper() == "L1":
            return nn.L1Loss()
        # PEMS만 L1 쓰던 기존 규칙도 유지하고 싶으면 아래 줄 활성화:
        if self.args.data == 'PEMS': return nn.L1Loss()
        return nn.MSELoss()


    def compute_ema_sequences(self, x, interval=0.01):
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
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.float(), batch_y.float(), batch_x_mark.float(), batch_y_mark.float()

        if 'PEMS' == self.args.data or 'Solar' == self.args.data:
            batch_x_mark = None
            batch_y_mark = None

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_mark = batch_x_mark.to(self.device) if batch_x_mark is not None else None
        batch_y_mark = batch_y_mark.to(self.device) if batch_y_mark is not None else None

        return batch_x, batch_y, batch_x_mark, batch_y_mark


    def process_batch_for_test(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_x.float(), batch_y.float(), batch_x_mark.float(), batch_y_mark.float()

        if 'PEMS' == self.args.data or 'Solar' == self.args.data:
            batch_x_mark = None
            batch_y_mark = None
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    # def sampling(self, x, x_mark, y_mark):
    #     batch_size = x.shape[0]
        
    #     x = x.to(self.device)
    #     x_mark = x_mark.to(self.device) if x_mark is not None else None
    #     y_mark = y_mark.to(self.device) if y_mark is not None else None

    #     output_t = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1)

    #     for alpha in self.alphas[1:]:
    #         output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))

    #     return output_t  # 최종 output
    
    def sampling(self, x, x_mark, y_mark):
        B = x.size(0)
        x = x.to(self.device)
        # 초기 예측: 마지막 관측을 pred_len만큼 반복
        y = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1)     # (B,T,C)

        # α 큰 → 작은 순으로 진행 (강한 열화부터 되돌린다는 느낌)
        for alpha in torch.flip(self.alphas[1:], dims=[0]):
            a = alpha.expand(B).to(self.device)
            # residual = fθ(current, x, α)
            residual = self.model(y, x, a)
            y = y + residual
        return y
    
    # 복원 학습용 열화 타깃 만들기
    def _degrade_y(self, batch_x, batch_y):
        """
        x_last와 y를 이어 붙여 EMA 체인을 만든 뒤, 임의의 α에서 열화된 y(z=T_α(y))를 뽑는다.
        반환: z (B,T,C), a (B,)
        """
        x_last = batch_x[:, -1:].to(self.device)                       # (B,1,C)
        seq = torch.cat([x_last, batch_y.to(self.device)], dim=1)      # (B,1+T,C)
        ema_all, _ = self.compute_ema_sequences(seq, interval=self.args.interval)  # (B,K,1+T,C)
        K = ema_all.size(1)

        # 각 샘플마다 α를 하나씩 랜덤 샘플
        idx = torch.randint(1, K, (seq.size(0),), device=seq.device)   # 1..K-1
        z = ema_all[torch.arange(seq.size(0), device=seq.device), idx, 1:, :]      # (B,T,C)
        a = self.alphas[idx].to(self.device)                           # (B,)
        return z, a

    
    
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
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.process_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                output = self.sampling(batch_x, batch_x_mark, batch_y_mark)
                loss = criterion(output, batch_y)
                total_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data,  vali_loader  = self._get_data(flag='val')
        test_data,  test_loader  = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # 이미지 저장 경로
        run_name = self._run_name(setting)
        train_fig_dir = os.path.join('./figs', run_name, 'train')
        os.makedirs(train_fig_dir, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion  = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                model_optim.zero_grad()

                # device 올리기
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.process_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )

                # === 한 스텝 복원 학습: z = T_α(y), pred = z + fθ(z, x, α) ===
                z, a = self._degrade_y(batch_x, batch_y)
                residual = self.model(z, batch_x, a)
                pred = z + residual

                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 안정화
                model_optim.step()
                scheduler.step()  # ★ 배치마다!

                # (선택) LR 로깅
                # wandb.log({"lr": scheduler.get_last_lr()[0]})

                # 100 스텝마다 그림 저장
                if i % 100 == 0:
                    with torch.no_grad():
                        x_np = batch_x.detach().cpu().numpy()
                        y_np = batch_y.detach().cpu().numpy()
                        p_np = pred.detach().cpu().numpy()

                        if train_data.scale and self.args.inverse:
                            shape = x_np.shape
                            x_np = train_data.inverse_transform(x_np.squeeze(0)).reshape(shape)

                        gt = np.concatenate((x_np[0, :, -1], y_np[0, :, -1]), axis=0)
                        pd = np.concatenate((x_np[0, :, -1], p_np[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(train_fig_dir, f"epoch_{epoch:03d}-iter_{i:05d}.pdf"))

                wandb.log({"epoch": epoch, "iteration": i, "train/loss": loss})
                pbar.set_postfix(loss=loss.item())

            # === 검증은 최종 파이프라인(sampling)으로 평가 ===
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.test(setting, test=False, epoch=epoch)

            wandb.log({"epoch": epoch, "val/loss": vali_loss, "test/loss": test_loss})
            print(f"Epoch: {epoch+1} | Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

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
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    if epoch is not None:
                        visual(gt, pd, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}.pdf"))
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

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        wandb.log({
            'test/mse': mse,
            'test/mae': mae,
        })
        return
