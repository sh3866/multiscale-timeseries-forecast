import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, visual
from utils.metrics import metric
from models import get_model

warnings.filterwarnings('ignore')


class Global_Loss(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # ==== α 그리드 안정화: linspace로 고정 개수 생성 ====
        steps = int(round(1.0 / self.args.interval))
        steps = max(1, steps)
        self.alphas = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)  # [0, ..., 1], 길이 A=steps+1
        self.num_steps = steps  # 스텝 수 K = A-1

        # AMP/Grad-clip
        self.use_amp = getattr(self.args, "use_amp", False)
        self.max_grad_norm = getattr(self.args, "max_grad_norm", 1.0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 시각화 빈도 제어
        self.plot_every = getattr(self.args, "plot_every", 10)

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices.replace(' ', '')
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Use Multi-GPU: {self.args.devices} (primary cuda:{self.args.gpu})')
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Use GPU: cuda:{self.args.gpu}')
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
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        # 데이터셋 명시보다 loss 플래그를 신뢰
        name = getattr(self.args, "loss", "MSE").upper()
        if name in ("L1", "MAE"):
            return nn.L1Loss()
        if name in ("HUBER", "SMOOTH_L1"):
            beta = getattr(self.args, "huber_beta", 1.0)
            return nn.SmoothL1Loss(beta=beta)
        return nn.MSELoss()

    # ==== α 역순 진행 헬퍼: 큰 α → 작은 α ====
    def _alpha_steps_desc(self):
        # α=0은 원시 타깃 쪽 끝. 복원 스케줄은 [1-Δ, ..., Δ] 사용
        return torch.flip(self.alphas[1:], dims=[0])  # 길이 K = A-1

    # ==== EMA 계산: α 축 브로드캐스트, 시간 루프만 ====
    def compute_ema_sequences(self, x):
        """
        x: (B, T, C), T에는 과거 마지막값 + 예측구간이 들어올 수 있음
        return:
          ema_outputs: (B, A, T, C)  where A = len(self.alphas)
          alpha_values: (B, A)
        """
        B, T, C = x.shape
        alphas = self.alphas.to(x.device)  # (A,)
        A = alphas.numel()

        ema = x[:, 0, :].unsqueeze(1).expand(B, A, C).contiguous()  # (B,A,C)
        outs = [ema.unsqueeze(2)]  # (B,A,1,C)

        a = alphas.view(1, A, 1)               # (1,A,1)
        one_minus_a = (1 - alphas).view(1, A, 1)

        for t in range(1, T):
            xt = x[:, t, :].unsqueeze(1)      # (B,1,C)
            ema = a * ema + one_minus_a * xt  # (B,A,C)
            outs.append(ema.unsqueeze(2))     # (B,A,1,C)

        ema_outputs = torch.cat(outs, dim=2)  # (B,A,T,C)
        alpha_values = alphas.unsqueeze(0).expand(B, A)  # (B,A)
        return ema_outputs, alpha_values

    # ==== 학습·검증 공용: 스텝 지도 데이터 구성 ====
    def process_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        EMA(y)에서 y_{alpha_k} -> y_{alpha_{k-1}} 지도 학습용 쌍을 구성.
        """
        _, seq_len, feature_dim = batch_x.shape
        T_pred = batch_y.shape[1]

        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = None if self.args.data in ('PEMS', 'Solar') else (batch_x_mark.float() if batch_x_mark is not None else None)
        batch_y_mark = None if self.args.data in ('PEMS', 'Solar') else (batch_y_mark.float() if batch_y_mark is not None else None)

        # 과거 마지막 값 + 미래구간 전체에 대해 EMA
        ema_all, alpha_values = self.compute_ema_sequences(
            torch.cat([batch_x[:, -1:].contiguous(), batch_y], dim=1)  # (B, 1+T_pred, C)
        )
        # t=1..T_pred만 사용
        ema_all = ema_all[:, :, 1:]  # (B, A, T_pred, C)

        A = self.alphas.numel()
        K = A - 1  # 단계 수

        # 입력 x를 K번 복제
        batch_x_rep = batch_x.unsqueeze(1).expand(-1, K, -1, -1).contiguous().view(-1, seq_len, feature_dim).to(self.device)
        # y_{alpha_k} (이전)와 y_{alpha_{k-1}} (타깃)
        batch_ema_y_prev = ema_all[:, 1:, :, :].contiguous().view(-1, T_pred, feature_dim).to(self.device)   # (B*K, T_pred, C)
        batch_ema_y_target = ema_all[:, :-1, :, :].contiguous().view(-1, T_pred, feature_dim).to(self.device) # (B*K, T_pred, C)
        # α 값도 길이를 K로 맞춰서 전개
        alpha_values_k = alpha_values[:, 1:].contiguous().view(-1).to(self.device)  # (B*K,)

        return batch_x_rep, batch_x_mark, batch_y_mark, batch_ema_y_prev, batch_ema_y_target, alpha_values_k

    # ==== 역방향 복원 샘플링 ====
    def sampling(self, x, x_mark, y_mark):
        """
        강 스무딩(상수에 가까운 상태)에서 역순 α 스케줄로 복원
        """
        batch_size = x.shape[0]
        # 🔧 dtype 통일
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(self.device).to(model_dtype)
        x_mark = x_mark.to(self.device).to(model_dtype) if x_mark is not None else None
        y_mark = y_mark.to(self.device).to(model_dtype) if y_mark is not None else None

        output_t = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1).to(model_dtype)
        for alpha in self._alpha_steps_desc():
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))
        return output_t

    # ==== 중간 단계까지의 모든 예측 텐서 반환(옵션) ====
    def sampling_with_intermediates_tensor(self, x, x_mark, y_mark):
        """
        (B, K, T_pred, C) 반환. 비용 큼. 필요할 때만 사용.
        """
        batch_size = x.shape[0]
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(self.device).to(model_dtype)
        x_mark = x_mark.to(self.device).to(model_dtype) if x_mark is not None else None
        y_mark = y_mark.to(self.device).to(model_dtype) if y_mark is not None else None

        output_t = x[:, -1].unsqueeze(1).repeat(1, self.args.pred_len, 1).to(model_dtype)
        preds = []
        for alpha in self._alpha_steps_desc():
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))
            preds.append(output_t)
        return torch.stack(preds, dim=1)  # (B, K, T_pred, C)

    # ==== 런 이름 ====
    def _run_name(self, fallback: str) -> str:
        try:
            if wandb.run is not None:
                return (wandb.run.name or wandb.run.id)
        except Exception:
            pass
        return fallback

    # ==== 검증 ====
    def vali(self, vali_data, vali_loader, criterion):
        """
        검증은 글로벌 복원 손실만 측정해 비용을 줄임.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.process_batch_for_test(batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred_y = self.sampling(batch_x, batch_x_mark, batch_y_mark)
                true_y = batch_y.to(self.device)
                loss = criterion(pred_y, true_y)
                total_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())
        avg = float(np.mean(total_loss)) if total_loss else 0.0
        self.model.train()
        return avg

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # 사진경로
        run_name = self._run_name(setting)
        train_fig_dir = os.path.join('./figs', run_name, 'train')
        os.makedirs(train_fig_dir, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (batch_x_raw, batch_y_raw, batch_x_mark_raw, batch_y_mark_raw) in pbar:
                model_optim.zero_grad(set_to_none=True)

                # dtype/device 정렬
                model_dtype = next(self.model.parameters()).dtype
                batch_x = batch_x_raw.to(self.device).to(model_dtype)         # (B, T_in, C)
                batch_y = batch_y_raw.to(self.device).to(model_dtype)         # (B, T_pred, C)

                # === 1) 랜덤 α 선택 =========================================
                # self.alphas: [0, a1, a2, ..., 1] 길이 A
                A = self.alphas.numel()
                start_idx = torch.randint(low=1, high=A, size=(1,), device=self.device).item()  # 1..A-1
                start_alpha = self.alphas[start_idx].item()

                # === 2) 해당 α에서의 corrupted 타깃 y_α 만들기 ==============
                # 입력: 과거 마지막값 + 미래정답 전체  → EMA 전체 계산
                ema_all, _ = self.compute_ema_sequences(
                    torch.cat([batch_x[:, -1:].contiguous(), batch_y], dim=1)  # (B, 1+T_pred, C)
                )  # (B, A, 1+T_pred, C)
                ema_all = ema_all[:, :, 1:]                                    # (B, A, T_pred, C)
                y_alpha = ema_all[:, start_idx, :, :].to(self.device)          # (B, T_pred, C)

                # === 3) start_alpha → 0 까지 역방향 복원 =====================
                # 알파 내림차순 중에서 start_alpha 이하만 사용
                alphas_desc = torch.flip(self.alphas[1:], dims=[0]).to(self.device).to(model_dtype)  # (A-1,)
                mask = alphas_desc <= start_alpha
                alphas_to_apply = alphas_desc[mask]  # [start_alpha, ..., 가장 작은 양의 α]

                output_t = y_alpha  # 초기 상태
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    for a in alphas_to_apply:
                        a_exp = a.expand(batch_x.size(0))
                        output_t = self.model(output_t, batch_x, a_exp)  # (B, T_pred, C)

                    # === 4) 최종 복원 결과와 GT 비교 =========================
                    loss = criterion(output_t, batch_y)

                # === 5) backward + clip + step ===============================
                self.scaler.scale(loss).backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(model_optim)
                self.scaler.update()

                pbar.set_postfix(loss=float(loss.item()))
                wandb.log({"epoch": epoch, "iteration": i, "train/total_loss": float(loss.item())})

                # === 6) 간단한 시각화(옵션) ==================================
                if hasattr(self, "plot_every") and self.plot_every and (i % self.plot_every == 0):
                    with torch.no_grad():
                        # 완전 상수(α=1)에서 시작한 샘플 하나 시각화
                        pred_y = self.sampling(batch_x[:1], None, None)  # sampling()이 내부에서 dtype/device 맞춤
                        true_y = batch_y[:1]
                        pred_np = pred_y.detach().cpu().numpy()
                        true_np = true_y.detach().cpu().numpy()
                        x_np = batch_x[:1].detach().cpu().numpy()
                        if train_data.scale and self.args.inverse:
                            pred_np = train_data.inverse_transform(pred_np.reshape(-1, pred_np.shape[-1])).reshape(pred_np.shape)
                            true_np = train_data.inverse_transform(true_np.reshape(-1, true_np.shape[-1])).reshape(true_np.shape)
                            x_np = train_data.inverse_transform(x_np.reshape(-1, x_np.shape[-1])).reshape(x_np.shape)
                        gt = np.concatenate((x_np[0, :, -1], true_np[0, :, -1]), axis=0)
                        pd = np.concatenate((x_np[0, :, -1], pred_np[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(train_fig_dir, f"epoch_{epoch}-{i}.pdf"))

            # === epoch 끝: 검증/테스트(샘플링 기반) ==========================
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.test(setting, test=False, epoch=epoch)
            wandb.log({"epoch": epoch, "val/loss": float(vali_loss), "test/loss": float(test_loss)})

            print(f"Epoch: {epoch+1} | Vali Loss: {vali_loss:.6f} Test Loss: {test_loss:.6f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        try:
            wandb.save(best_model_path)
        except Exception:
            pass
        return self.model


    def process_batch_for_test(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        model_dtype = next(self.model.parameters()).dtype
        batch_x = batch_x.to(model_dtype)
        batch_y = batch_y.to(model_dtype)
        if self.args.data in ('PEMS', 'Solar'):
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.to(model_dtype) if batch_x_mark is not None else None
            batch_y_mark = batch_y_mark.to(model_dtype) if batch_y_mark is not None else None
        return batch_x, batch_y, batch_x_mark, batch_y_mark

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

                preds.append(outputs)
                trues.append(batch_y)

                # 저장 빈도 제한
                if self.plot_every and i % self.plot_every == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input_np[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(test_fig_dr, f"epoch_{str(epoch)}-{str(i)}.pdf" if epoch is not None else f"{str(i)}.pdf"))

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
        wandb.log({'test/mse': mse, 'test/mae': mae})
        return
