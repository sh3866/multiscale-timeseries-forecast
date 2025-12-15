import os
import warnings
import re

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

import torch.nn.functional as F

warnings.filterwarnings('ignore')


class Test(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        
        self.enable_mu_predictor = (getattr(self.args, "use_ma_start", 0) == 2)
        

        # ==== α 그리드 안정화: linspace로 고정 개수 생성 ====
        steps = int(round(1.0 / self.args.interval))
        steps = max(1, steps)
        self.alphas = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)  # [0, ..., 1], 길이 A=steps+1
        self.num_steps = steps  # 스텝 수 K = A-1
    
        # =========================================================================================
        # power = getattr(self.args, "alpha_power", 0.3)  # 0.3 ~ 0.7 추천, 설정 없으면 0.3

        # x = torch.linspace(0.0, 1.0, steps + 1)
        # self.alphas = x.pow(power).to(torch.float32)  # skew 적용

        # self.num_steps = steps

        # print("\n[Alpha Grid Info]")
        # print("interval:", self.args.interval)
        # print("steps:", steps)
        # print("power:", power)
        # print("alphas:", self.alphas.cpu().numpy())  # 실제 값 확인
        # =========================================================================================
        # =========================================================================================

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

    # ==== MATSD 스타일 MA-diffusion smoothing ====
    def compute_ema_sequences(self, x):
        """
        MATSD 방식으로 여러 단계의 moving-average diffusion 상태 x_t 를 생성.

        x: (B, T, C)
           - T에는 "과거 마지막 값 + 예측 구간" 등을 자유롭게 넣어도 됨.
        return:
          ema_outputs:  (B, A, T, C)   # 각 alpha step마다 MA로 스무딩된 시계열
          alpha_values: (B, A)         # self.alphas 복제
        """

        # ------------------ 0. 내부 helper (이 함수 안에서만 사용) ------------------
        def _get_factors(n: int):
            """
            MATSD repo와 비슷하게 n의 약수들을 오름차순으로 반환. 1과 n 포함.
            kernel size 후보가 된다.
            """
            f = list(
                set(
                    factor
                    for i in range(2, int(n ** 0.5) + 1)
                    if n % i == 0
                    for factor in (i, n // i)
                )
            )
            f.sort()
            f.append(n)
            return [1] + f  # 항상 1로 시작 (거의 identity)

        def _build_transition_matrix(seq_length: int, kernel_size: int,
                                     device, dtype):
            """
            MATSD 그림과 동일한 방식으로 time-domain MA transition matrix K (T,T) 생성.

            1) kernel_size 길이의 moving-average kernel을 슬라이딩하면서
               각 column에 unroll
            2) column 축(n_windows)을 time step 길이(seq_length)로 interpolate
            3) (T, T) 정사각 행렬 반환 (row = output time, col = input time)
            """
            stride = 1
            # (T, n_windows)
            K = torch.zeros(
                seq_length,
                int((seq_length - kernel_size) / stride + 1),
                device=device,
                dtype=dtype,
            )
            start = 0
            for i in range(K.shape[1]):
                end = start + kernel_size
                K[start:end, i] = 1.0 / kernel_size
                start += stride

            # Unroll 된 kernel들을 time step 방향으로 interpolate
            # 현재 K: (T, n_windows) -> (1, T, n_windows) 형식으로 바꿔서
            # 마지막 축(n_windows)을 seq_length로 리샘플링.
            K = K.unsqueeze(0)  # (1, T, n_windows)  # N,C,L 형식에서 C=T, L=n_windows
            mode = "nearest-exact" if stride == 1 else "linear"
            K = torch.nn.functional.interpolate(
                K, size=seq_length, mode=mode
            ).squeeze(0)            # (T, T)
            K = K.T                  # (T, T)  row = output time, col = input time
            return K

        # ------------------ 1. 기본 셋업 ------------------
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # α grid: (A,)  — 네가 이미 interval / power 로 만든 그 값
        alphas = self.alphas.to(device=device, dtype=dtype)
        A = alphas.numel()          # diffusion step 개수(=alpha 개수)

        # ------------------ 2. kernel size bank 만들기 ------------------
        # T의 약수 기반으로 여러 kernel size 후보 k_i 생성
        #   k=1   -> 거의 identity
        #   k=T   -> 가장 강한 smoothing
        factor_list = _get_factors(T)      # 예: [1, 2, 4, 8, ..., T]
        num_factors = len(factor_list)

        # 각 kernel_size에 대한 (T,T) transition matrix K_i 계산
        Ks = []
        for ksize in factor_list:
            Ks.append(
                _build_transition_matrix(
                    seq_length=T,
                    kernel_size=ksize,
                    device=device,
                    dtype=dtype,
                )
            )
        Ks = torch.stack(Ks, dim=0)        # (F, T, T),  F=num_factors

        # ------------------ 3. diffusion step(α)→K_t 매핑 (Interp. on {K_i}) ------------------
        # 원본 MATSD처럼 여러 kernel 사이를 diffusion step 방향으로 interpolation.
        #   alphas[0] ~ alphas[-1] 를 [0, F-1] 구간에 선형 매핑해서
        #   인접한 두 K_i 사이를 convex combination.
        if num_factors == 1:
            Ks_alpha = Ks.repeat(A, 1, 1)  # corner case: 약수가 하나뿐인 경우
        else:
            alpha_min, alpha_max = alphas[0], alphas[-1]
            # 0~1 로 정규화된 step 위치
            alpha_norm = (alphas - alpha_min) / (alpha_max - alpha_min + 1e-8)  # (A,)
            pos = alpha_norm * (num_factors - 1)                                 # (A,)

            idx0 = torch.floor(pos).long()                                       # (A,)
            idx1 = torch.clamp(idx0 + 1, max=num_factors - 1)                    # (A,)
            w1 = (pos - idx0).view(A, 1, 1)                                      # (A,1,1)
            w0 = 1.0 - w1

            # K_t = (1-w)*K_idx0 + w*K_idx1   ← MATSD 그림의 "Interp. on diffusion steps {K_i}"
            Ks_alpha = w0 * Ks[idx0] + w1 * Ks[idx1]                             # (A, T, T)

        # 이제 Ks_alpha[a] 가 원본 MATSD에서 noise_schedule["alphas"][t]에 해당하는 K_t.

        # ------------------ 4. 각 step t에서 x_t = K_t @ x 계산 ------------------
        # x: (B, T, C) → (B*C, T) → (T, B*C)
        x_flat = x.permute(0, 2, 1).reshape(-1, T)   # (B*C, T)
        x_flat = x_flat.t()                          # (T, B*C)

        # outs_alpha[a] = K_t[a] @ x_flat
        #   Ks_alpha: (A, T, T)
        #   x_flat:   (T, B*C)
        # ⇒ outs_alpha: (A, T, B*C)
        outs_alpha = torch.einsum('aij,jk->aik', Ks_alpha, x_flat)

        # 다시 (B, A, T, C) 로 reshape
        outs_alpha = outs_alpha.permute(2, 0, 1)     # (B*C, A, T)
        outs_alpha = outs_alpha.reshape(B, C, A, T)  # (B, C, A, T)
        ema_outputs = outs_alpha.permute(0, 2, 3, 1) # (B, A, T, C)
        
        # === 4.5 Drift term 추가 (마지막 값 attractor) ===

        # (B, 1, 1, C)
        last_val = x[:, :1, :]  
        mean_val = x.mean(dim=1, keepdim=True)
        n = mean_val - last_val   # (B, 1, 1, C)
        n = n.unsqueeze(2)  

        # (A,) → (1,A,1,1)
        step_factors = torch.linspace(0, 1, A, device=device, dtype=dtype)
        step_factors = step_factors.view(1, A, 1, 1)

        # n: (B,1,1,C) → (B,1,1,C) 그대로 broadcast됨
        drift = step_factors * n              # (B,A,1,C)

        # subtract drift
        ema_outputs = ema_outputs - drift

        # ------------------ 5. α 값 브로드캐스트 ------------------
        alpha_values = alphas.unsqueeze(0).expand(B, A)  # (B, A)

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
        # ==== 역방향 복원 샘플링 ====
    def sampling(self, x, x_mark, y_mark, y=None, use_ma_start=0):
        """
        use_ma_start:
          0: 마지막 관측값 상수에서 시작 (기존 방식)
          1: GT future EMA 최강 스무딩 상수에서 시작 (peeking)
          2: μ/σ predictor 기반 상수에서 시작 (논문식)
        x: (B, T_in, C)
        y: (B, T_pred, C)  - 모드 1에서만 필요
        """
        batch_size = x.shape[0]
        model_dtype = next(self.model.parameters()).dtype

        # bool로 들어와도 안전하게 int로 변환
        if isinstance(use_ma_start, bool):
            mode = 1 if use_ma_start else 0
        else:
            mode = int(use_ma_start)

        # dtype / device 정렬
        x = x.to(self.device).to(model_dtype)
        x_mark = x_mark.to(self.device).to(model_dtype) if x_mark is not None else None
        y_mark = y_mark.to(self.device).to(model_dtype) if y_mark is not None else None

        T_pred = self.args.pred_len

        # =======================================================
        # 시작 상태 결정
        # =======================================================
        if mode == 2:
            # DiT 내부 μ-head로 μ 예측 (과거 x만 사용)
            mu_hat = self.model.predict_mu(x)          # (B, C)
            output_t = mu_hat.unsqueeze(1).expand(batch_size, T_pred, -1)



        elif mode == 1 and (y is not None):
            # --- GT 기반 EMA peeking (지금까지 쓰던 방식) ---
            y = y.to(self.device).to(model_dtype)                   # (B, T_pred, C)
            ema_all, _ = self.compute_ema_sequences(y)              # (B, A, T_pred, C)
            ema_max_smooth = ema_all[:, -1, :, :]                   # (B, T_pred, C)
            const_val = ema_max_smooth.mean(dim=1, keepdim=True)    # (B, 1, C)
            output_t = const_val.repeat(1, T_pred, 1)               # (B, T_pred, C)

        else:
            # --- 완전 기존 default: 마지막 관측값 상수 ---
            output_t = x[:, -1].unsqueeze(1).repeat(1, T_pred, 1).to(model_dtype)

        # =======================================================
        # 역방향 α 스케줄 복원
        # =======================================================
        for alpha in self._alpha_steps_desc():
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))
        return output_t




    # ==== 중간 단계까지의 모든 예측 텐서 반환(옵션) ====
        # ==== 중간 단계까지의 모든 예측 텐서 반환(옵션) ====
    def sampling_with_intermediates_tensor(
        self,
        x,
        x_mark,
        y_mark,
        y=None,
        use_ma_start=0,
        return_init: bool = False,
    ):
        """
        (B, K, T_pred, C) 반환.
        return_init=True 이면 (init_output_t, preds_all) 같이 반환.

        use_ma_start:
          0: last obs 상수
          1: EMA(GT) 상수
          2: μ/σ predictor 상수
        """
        batch_size = x.shape[0]
        model_dtype = next(self.model.parameters()).dtype

        # bool로 들어와도 안전하게 int로 변환
        if isinstance(use_ma_start, bool):
            mode = 1 if use_ma_start else 0
        else:
            mode = int(use_ma_start)

        # dtype / device 정렬
        x = x.to(self.device).to(model_dtype)
        x_mark = x_mark.to(self.device).to(model_dtype) if x_mark is not None else None
        y_mark = y_mark.to(self.device).to(model_dtype) if y_mark is not None else None

        T_pred = self.args.pred_len

        # ---- 초기 상태 결정 (sampling 과 동일) ----
        if mode == 2:
            # DiT 내부 μ-head로 μ 예측
            mu_hat = self.model.predict_mu(x)          # (B, C)
            output_t = mu_hat.unsqueeze(1).expand(batch_size, T_pred, -1)


        elif mode == 1 and (y is not None):
            y = y.to(self.device).to(model_dtype)
            ema_all, _ = self.compute_ema_sequences(y)      # (B, A, T_pred, C)
            ema_max_smooth = ema_all[:, -1, :, :]
            const_val = ema_max_smooth.mean(dim=1, keepdim=True)
            output_t = const_val.repeat(1, T_pred, 1)
        else:
            output_t = x[:, -1].unsqueeze(1).repeat(1, T_pred, 1).to(model_dtype)

        # ★ 모델에 들어가기 직전 초기 상태 저장
        init_out = output_t.clone()

        preds = []
        for alpha in self._alpha_steps_desc():
            output_t = self.model(output_t, x, alpha.expand(batch_size).to(self.device))
            preds.append(output_t)

        preds_all = torch.stack(preds, dim=1)  # (B, K, T_pred, C)

        if return_init:
            return init_out, preds_all
        return preds_all



    # ==== 런 이름 ====
    def _run_name(self, fallback: str) -> str:
        try:
            if wandb.run is not None:
                return (wandb.run.name or wandb.run.id)
        except Exception:
            pass
        return fallback
    
    def _run_numeric_suffix(self, fallback: str) -> str:
        """
        wandb run name(예: 'amber-planet-193')에서
        맨 뒤 숫자('193')만 떼서 반환.
        숫자가 없으면 fallback 그대로 사용.
        """
        base = None
        try:
            if wandb.run is not None:
                base = (wandb.run.name or wandb.run.id)
        except Exception:
            pass
        if base is None:
            base = fallback

        m = re.search(r'(\d+)$', str(base))
        return m.group(1) if m else str(base)

    def _fig_root(self, fallback: str) -> str:
        """
        figs/<fig_tag(optional)>/<numeric_suffix> 까지의 경로를 반환.
        fig_tag가 없으면 figs/<numeric_suffix> 형태.
        """
        suffix = self._run_numeric_suffix(fallback)
        fig_tag = getattr(self.args, "fig_tag", None)
        if fig_tag:
            return os.path.join("./figs", fig_tag, suffix)
        else:
            return os.path.join("./figs", suffix)

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
                
                start_mode = getattr(self.args, "use_ma_start", 0)
                pred_y = self.sampling(batch_x, batch_x_mark, batch_y_mark, batch_y, use_ma_start=start_mode)
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
        # run_name = self._run_name(setting)
        # train_fig_dir = os.path.join('./figs', run_name, 'train')
        fig_root = self._fig_root(setting)              # figs/(fig_tag)/숫자
        train_fig_dir = os.path.join(fig_root, 'train') # .../train
        os.makedirs(train_fig_dir, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)


        criterion = self._select_criterion()
        
        # ==============================================================
        #  학습 시작 전에: test 샘플 10개 EMA smoothing 전체 시각화
        # ==============================================================

        # train_data, train_loader = self._get_data(flag='train')
        # batch_x, batch_y, _, _ = next(iter(train_loader))
        # batch_x = batch_x[:10].to(self.device)   # (10, T_in, C)
        # batch_y = batch_y[:10].to(self.device)   # (10, T_out, C)

        # # [past_last, future] 연결
        # xy = torch.cat([batch_x[:, -1:], batch_y], dim=1)  # (10, 1+T_out, C)

        # # EMA 전체 생성
        # ema_all, _ = self.compute_ema_sequences(xy)  # (10, A, 1+T_out, C)
        # ema_all = ema_all[:, :, 1:, :]               # (10, A, T_out, C)

        # B, A, T_pred, C = ema_all.shape

        # fig_root = self._fig_root(setting)
        # plot_root = os.path.join(fig_root, 'ema_all_init')   # 초기 EMA 시각화 폴더
        # os.makedirs(plot_root, exist_ok=True)

        # for s in range(B):  # 10개
        #     sample_dir = os.path.join(plot_root, f'sample{s+1}')
        #     os.makedirs(sample_dir, exist_ok=True)

        #     true = batch_y[s, :, -1].detach().cpu().numpy()

        #     for idx in range(A):
        #         alpha = float(self.alphas[idx].item())
        #         pred_alpha = ema_all[s, idx, :, -1].detach().cpu().numpy()

        #         plt.figure()
        #         plt.plot(true, label='Original Future')
        #         plt.plot(pred_alpha, label=f'EMA α={alpha:.2f}')
        #         plt.legend()
        #         plt.title(f'[Init EMA] Sample {s+1} (α={alpha:.2f})')
        #         plt.savefig(os.path.join(sample_dir, f'ema_alpha_{alpha:.2f}.pdf'))
        #         plt.close()

        # print(">>> 초기 EMA smoothing 10개 시각화 완료")
        
        
        # ========================================================
        lambda_traj = getattr(self.args, "lambda_traj", 1.0)
        lambda_end  = getattr(self.args, "lambda_end", 1.0)
        lambda_mu   = getattr(self.args, "lambda_mu", 0.0)
        # ========================================================
        
        printed = False

        for epoch in range(self.args.train_epochs):
            self.model.train()
            
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (batch_x_raw, batch_y_raw, batch_x_mark_raw, batch_y_mark_raw) in pbar:
                
                if not printed:
                    print(">>> batch_x shape:", batch_x_raw.shape)
                    printed = True

                
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
                ema_all = ema_all.to(model_dtype)

                A = self.alphas.numel()
                # 1..A-1 범위에서 2개 랜덤 선택
                idx1 = torch.randint(1, A, (1,), device=self.device).item()
                idx2 = torch.randint(1, A, (1,), device=self.device).item()

                # 큰 쪽이 start / 작은 쪽이 end
                start_idx = max(idx1, idx2)
                end_idx = min(idx1, idx2)

                # 최소 1차이는 나게 하기 (동일 인덱스 방지)
                if start_idx == end_idx:
                    if start_idx < A-1:
                        start_idx += 1
                    else:
                        end_idx -= 1

                # teacher에서 start step 상태 (y_{alpha_start})
                y_alpha = ema_all[:, start_idx, :, :]                          # (B, T_pred, C)

                # === 3) start_idx → 0 까지 step-wise trajectory 학습 ============

                output_t = y_alpha  # 초기 상태: teacher EMA at start_idx
                traj_loss = 0.0
                num_traj_steps = 0

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # === 2) start_idx → end_idx+1 까지만 반복 ===
                    for alpha_idx in range(start_idx, end_idx, -1):
                        a_val = self.alphas[alpha_idx].to(self.device).to(model_dtype)
                        a_exp = a_val.expand(batch_x.shape[0])

                        # predict α_{k-1}
                        output_t = self.model(output_t, batch_x, a_exp)

                        # teacher EMA target
                        teacher_next = ema_all[:, alpha_idx - 1, :, :]
                        step_loss = criterion(output_t, teacher_next)
                        traj_loss += step_loss
                        num_traj_steps += 1

                    if num_traj_steps > 0:
                        traj_loss /= num_traj_steps

                    # === 3) end loss를 "end_idx" 에서의 teacher EMA 또는 GT 비교로 선택 ===
                    if end_idx == 0:
                        # end_idx = 0 → 진짜 Ground Truth 비교
                        end_loss = criterion(output_t, batch_y)
                    else:
                        # 중간 end이면 Teacher EMA 비교
                        end_loss = criterion(output_t, ema_all[:, end_idx, :, :])

                    # loss = lambda_traj * traj_loss + lambda_end * end_loss
                    
                    # === μ loss (Mean predictor) =======================================
                    mu_loss = torch.tensor(0.0, device=self.device, dtype=model_dtype)

                    if self.enable_mu_predictor:
                        # DiT 내부 μ-head 사용
                        mu_hat = self.model.predict_mu(batch_x)   # (B, C)
                        true_mu = batch_y.mean(dim=1)             # (B, C)
                        mu_loss = criterion(mu_hat, true_mu)


                    # lambda_mu 는 mode==2에서만 활성화
                    effective_lambda_mu = lambda_mu if self.enable_mu_predictor else 0.0

                    loss = (
                        lambda_traj * traj_loss
                        + lambda_end * end_loss
                        + effective_lambda_mu * mu_loss
                    )



                # === 5) backward + clip + step ===============================
                self.scaler.scale(loss).backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(model_optim)
                self.scaler.update()

                pbar.set_postfix(loss=float(loss.item()))
                # wandb.log({"epoch": epoch, "iteration": i, "train/total_loss": float(loss.item())})
                wandb.log({
                    "epoch": epoch,
                    "iteration": i,
                    "train/total_loss": float(loss.item()),
                    "train/traj_loss": float(traj_loss.item()),
                    "train/end_loss": float(end_loss.item()),
                    "train/mu_loss": float(mu_loss.item()),
                })


                # === 6) 간단한 시각화(옵션) ==================================
                if hasattr(self, "plot_every") and self.plot_every and (i % self.plot_every == 0):
                    with torch.no_grad():
                        # 완전 상수(α=1)에서 시작한 샘플 하나 시각화
                        start_mode = getattr(self.args, "use_ma_start", 0)
                        pred_y = self.sampling(batch_x[:1], None, None,
                                               batch_y[:1], use_ma_start=start_mode)

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
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []

        # run_name = self._run_name(setting)
        # test_fig_dr = os.path.join('./figs', run_name, 'test')
        fig_root = self._fig_root(setting)
        test_fig_dr = os.path.join(fig_root, 'test')
        os.makedirs(test_fig_dr, exist_ok=True)

        # === 추가: alpha 결과 루트 디렉터리 및 알파 값(내림차순) 미리 준비 ===
        # alpha_root_dir = os.path.join('./figs', run_name, 'alpha')
        alpha_root_dir = os.path.join(fig_root, 'alpha')
        os.makedirs(alpha_root_dir, exist_ok=True)
        alpha_values_desc = self._alpha_steps_desc()        # Tensor (K,)
        alpha_values_desc_np = alpha_values_desc.detach().cpu().numpy()

        self.model.eval()
    
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark = \
                    self.process_batch_for_test(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # sampling (최종 결과)
                start_mode = getattr(self.args, "use_ma_start", 0)
                outputs = self.sampling(batch_x, batch_x_mark, batch_y_mark,
                                        batch_y, use_ma_start=start_mode)


                # numpy 버전 (metrics / 기존 plot용)
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()

                preds.append(outputs_np)
                trues.append(batch_y_np)

                # 저장 빈도 제한
                if self.plot_every and i % self.plot_every == 0:
                    # ----- 1) 기존 test sample 그림 저장 -----
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(
                            input_np.squeeze(0)
                        ).reshape(shape)

                    gt = np.concatenate(
                        (input_np[0, :, -1], batch_y_np[0, :, -1]),
                        axis=0,
                    )
                    pd = np.concatenate(
                        (input_np[0, :, -1], outputs_np[0, :, -1]),
                        axis=0,
                    )
                    sample_fname = (
                        f"epoch_{str(epoch)}-{str(i)}.pdf"
                        if epoch is not None
                        else f"{str(i)}.pdf"
                    )
                    visual(
                        gt,
                        pd,
                        os.path.join(test_fig_dr, sample_fname),
                    )

                    # ----- 2) 추가: 이 test sample 에 대한 모든 α-step 결과 저장 -----
                    sample_name_no_ext = sample_fname.replace(".pdf", "")
                    sample_alpha_dir = os.path.join(alpha_root_dir, sample_name_no_ext)
                    os.makedirs(sample_alpha_dir, exist_ok=True)

                    # 이 sample(배치의 첫 번째) 기준으로 중간 step 전부 얻기
                    batch_x_1 = batch_x[0:1]                      # (1, T_in, C)
                    bx_mark_1 = batch_x_mark[0:1] if batch_x_mark is not None else None
                    by_mark_1 = batch_y_mark[0:1] if batch_y_mark is not None else None
                    batch_y_1 = batch_y[0:1]                      # (1, T_pred, C)

                    # ★ 초기 상태(init_out_1) + 모든 step(preds_all) 한 번에 가져오기
                    init_out_1, preds_all = self.sampling_with_intermediates_tensor(
                        batch_x_1, bx_mark_1, by_mark_1,
                        y=batch_y_1, use_ma_start=start_mode,
                        return_init=True,
                    )  # init_out_1: (1, T_pred, C), preds_all: (1, K, T_pred, C)

                    preds_all_np = preds_all.squeeze(0).detach().cpu().numpy()  # (K, T_pred, C)
                    init_future_1 = init_out_1[0, :, -1].detach().cpu().numpy() # (T_pred,)

                    # history / true future (이미 inverse 된 input_np, batch_y_np 사용)
                    history_1 = input_np[0, :, -1]          # (T_in,)
                    true_future_1 = batch_y_np[0, :, -1]    # (T_pred,)

                    # ---------- (a) 초기 상태: 모델 들어가기 전 output_t 그대로 ----------
                    gt_series_init = np.concatenate(
                        (history_1, true_future_1), axis=0
                    )
                    pd_series_init = np.concatenate(
                        (history_1, init_future_1), axis=0
                    )

                    plt.figure()
                    plt.plot(gt_series_init, label="Ground Truth")
                    plt.plot(
                        pd_series_init,
                        label="Init (before model)",
                    )
                    plt.legend()
                    plt.title(f"{sample_name_no_ext} | init (before step 1)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(sample_alpha_dir, "alpha_000.pdf"))
                    plt.close()

                    # ---------- (b) 이후 각 α-step 별 결과 ----------
                    num_steps = preds_all_np.shape[0]  # = K
                    for k in range(num_steps):
                        alpha_val = float(alpha_values_desc_np[k])
                        pred_step = preds_all_np[k, :, -1]  # (T_pred,)

                        gt_series = np.concatenate(
                            (history_1, true_future_1), axis=0
                        )
                        pd_series = np.concatenate(
                            (history_1, pred_step), axis=0
                        )

                        step_idx = k + 1  # step 1부터

                        plt.figure()
                        plt.plot(gt_series, label="Ground Truth")
                        plt.plot(
                            pd_series,
                            label=f"Pred (step {step_idx}, alpha={alpha_val:.3f})",
                        )
                        plt.legend()
                        plt.title(
                            f"{sample_name_no_ext} | step {step_idx} (alpha={alpha_val:.3f})"
                        )
                        plt.tight_layout()

                        alpha_fname = f"alpha_{step_idx:03d}.pdf"
                        plt.savefig(os.path.join(sample_alpha_dir, alpha_fname))
                        plt.close()

        # ===== 원래 하단 metrics 부분 그대로 유지 =====
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