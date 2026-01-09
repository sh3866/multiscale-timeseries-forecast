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
                device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES로 제한했으므로 항상 0
                print(f'Use GPU: cuda:0 (physical GPU: {self.args.gpu})')
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

        # 데이터셋 정보 출력 (디버깅용)
        if flag == 'train':
            print(f"\n{'='*60}")
            print(f"Dataset Loading Information ({flag})")
            print(f"{'='*60}")
            print(f"Data type: {self.args.data}")
            print(f"Root path: {self.args.root_path}")
            print(f"Data path: {self.args.data_path}")
            print(f"Features: {self.args.features}")
            print(f"Target: {self.args.target}")
            print(f"Seq len: {self.args.seq_len}")
            print(f"Pred len: {self.args.pred_len}")
            print(f"Dataset size: {len(data_set)}")
            print(f"Dataset class: {type(data_set).__name__}")

            # 첫 번째 샘플 확인
            sample_x, sample_y, _, _ = data_set[0]
            print(f"Sample X shape: {sample_x.shape}")
            print(f"Sample Y shape: {sample_y.shape}")
            print(f"Sample X range: [{sample_x.min():.4f}, {sample_x.max():.4f}]")
            print(f"Sample Y range: [{sample_y.min():.4f}, {sample_y.max():.4f}]")

            # Scaler 정보 확인
            if hasattr(data_set, 'scaler'):
                print(f"\nScaler info:")
                print(f"  Mean: {data_set.scaler.mean_[0]:.4f}")
                print(f"  Std:  {data_set.scaler.scale_[0]:.4f}")

            # Alpha grid 정보
            print(f"\nAlpha grid info:")
            print(f"  Number of alphas: {len(self.alphas)}")
            print(f"  Alpha range: [{self.alphas[0]:.4f}, {self.alphas[-1]:.4f}]")
            print(f"  Alpha interval: {self.args.interval}")

            # Factor 정보 (96의 약수)
            T = sample_y.shape[0]
            factors = []
            for i in range(2, int(T ** 0.5) + 1):
                if T % i == 0:
                    factors.append(i)
                    if i != T // i:
                        factors.append(T // i)
            factors = sorted(set(factors))
            factors = [1] + factors + [T] if T not in factors else [1] + factors
            print(f"\nPred_len factors (T={T}): {factors}")
            print(f"Number of kernel sizes: {len(factors)}")

            print(f"{'='*60}\n")

        return data_set, data_loader

    def _select_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)

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

    def _fig_root(self, fallback: str) -> str:
        """
        figs/<fig_tag(optional)>/<run_name> 까지의 경로를 반환.
        fig_tag가 없으면 figs/<run_name> 형태.

        run_name 예시:
        - ETTm1_96_S
        - electricity_192_M_parameter_test
        - weather_336_S_ablation
        """
        run_name = self._run_name(fallback)
        fig_tag = getattr(self.args, "fig_tag", None)
        if fig_tag:
            return os.path.join("./figs", fig_tag, run_name)
        else:
            return os.path.join("./figs", run_name)

    # ==== 검증 ====
    def vali(self, vali_data, vali_loader, criterion):
        """
        검증은 글로벌 복원 손실만 측정 (standardized space에서 계산).
        test와 동일하게 전체 prediction을 모은 후 MSE 계산.
        """
        preds = []
        trues = []

        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self.process_batch_for_test(batch_x, batch_y, batch_x_mark, batch_y_mark)

                start_mode = getattr(self.args, "use_ma_start", 0)
                pred_y = self.sampling(batch_x, batch_x_mark, batch_y_mark, batch_y, use_ma_start=start_mode)

                preds.append(pred_y.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        # 전체 prediction에 대해 MSE 계산 (test와 동일)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mse = np.mean((preds - trues) ** 2)

        self.model.train()
        return mse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Checkpoint 경로: setting에 이미 run.id가 포함되어 있으면 (sweep) 그대로 사용
        # 아니면 (기존 sh) hyperparameter를 붙여서 고유한 경로 생성
        if '/' in setting:
            # sweep: setting = "ETTm2_96_96_S/run_id"
            path = os.path.join(self.args.checkpoints, setting)
        else:
            # 기존 sh: hyperparameter 추가
            hp_suffix = f"_hd{self.args.hidden_dim}_nh{self.args.num_heads}_nb{self.args.num_dit_block}"
            path = os.path.join(self.args.checkpoints, setting + hp_suffix)
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

                # === 1) EMA 전체 계산 =========================================
                # 입력: 과거 마지막값 + 미래정답 전체  → EMA 전체 계산
                ema_all, _ = self.compute_ema_sequences(
                    torch.cat([batch_x[:, -1:].contiguous(), batch_y], dim=1)  # (B, 1+T_pred, C)
                )  # (B, A, 1+T_pred, C)
                ema_all = ema_all[:, :, 1:]                                    # (B, A, T_pred, C)
                ema_all = ema_all.to(model_dtype)

                # === 2) Random two-step loss ===================================
                # 랜덤으로 start_idx, end_idx 선택 (start_idx > end_idx)
                # start_idx: 큰 alpha (smooth), end_idx: 작은 alpha (원본에 가까움)
                A = self.alphas.numel()  # alpha grid 크기

                # 최소 1스텝 이상의 간격 보장: start_idx ∈ [1, A-1], end_idx ∈ [0, start_idx-1]
                start_idx = torch.randint(1, A, (1,)).item()
                end_idx = torch.randint(0, start_idx, (1,)).item()

                num_steps = start_idx - end_idx  # 복원해야 할 스텝 수

                # 시작 상태: start_idx의 EMA 정답
                output_t = ema_all[:, start_idx, :, :].clone()  # (B, T_pred, C)

                # 정답: end_idx의 EMA (end_idx=0이면 원본 GT)
                if end_idx == 0:
                    target_end = batch_y  # 원본 GT
                else:
                    target_end = ema_all[:, end_idx, :, :]

                # === 3) Traj Loss: 각 스텝 detach로 끊어서 독립 학습 ===
                traj_loss = torch.tensor(0.0, device=self.device, dtype=model_dtype)
                num_traj_steps = 0

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Traj Loss 계산 (detach로 각 스텝 독립)
                    output_traj = ema_all[:, start_idx, :, :].clone()  # 시작점
                    for alpha_idx in range(start_idx, end_idx, -1):
                        a_val = self.alphas[alpha_idx].to(self.device).to(model_dtype)
                        a_exp = a_val.expand(batch_x.shape[0])

                        # 모델 예측
                        output_traj = self.model(output_traj, batch_x, a_exp)

                        # Traj Loss: 현재 예측 vs 다음 alpha의 EMA 정답
                        if alpha_idx - 1 == 0:
                            target_ema = batch_y  # 원본 GT
                        else:
                            target_ema = ema_all[:, alpha_idx - 1, :, :]
                        traj_loss = traj_loss + criterion(output_traj, target_ema)
                        num_traj_steps += 1

                        # 다음 스텝으로 넘어갈 때 gradient 끊기
                        output_traj = output_traj.detach()

                    # Traj Loss 평균
                    if num_traj_steps > 0:
                        traj_loss = traj_loss / num_traj_steps

                    # === 4) End Loss: 전체 trajectory gradient 흐름 (detach 없음) ===
                    output_end = ema_all[:, start_idx, :, :].clone()  # 시작점 (새로 시작)
                    for alpha_idx in range(start_idx, end_idx, -1):
                        a_val = self.alphas[alpha_idx].to(self.device).to(model_dtype)
                        a_exp = a_val.expand(batch_x.shape[0])
                        output_end = self.model(output_end, batch_x, a_exp)
                        # detach 없음: gradient가 전체 trajectory를 통해 흐름

                    # End Loss: 최종 예측 vs end_idx 정답
                    end_loss = criterion(output_end, target_end)

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
            # train과 동일한 경로 구성: sweep vs 기존 sh 분기
            if '/' in setting:
                # sweep: setting = "ETTm2_96_96_S/run_id"
                ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            else:
                # 기존 sh: hyperparameter 추가
                hp_suffix = f"_hd{self.args.hidden_dim}_nh{self.args.num_heads}_nb{self.args.num_dit_block}"
                ckpt_path = os.path.join(self.args.checkpoints, setting + hp_suffix, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt_path))

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

        # TimeMixer 방식: standardized space에서 메트릭 계산
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        wandb.log({'test/mse': mse, 'test/mae': mae})

        # ========== 전체 feature 시각화 (랜덤 샘플) ==========
        self._visualize_all_features(preds, trues, test_data, test_loader, setting, epoch=epoch)

        # ================================================================================
        # [Alpha-step별 MSE 분석] - 주석 해제하면 각 스텝별 MSE 계산 및 막대그래프 저장
        # 시간이 오래 걸릴 수 있음 (전체 test set에 대해 모든 중간 스텝 계산)
        # ================================================================================
        self._compute_alpha_step_mse(test_data, test_loader, setting, epoch=epoch)
        # ================================================================================

        return

    def _compute_alpha_step_mse(self, test_data, test_loader, setting, epoch=None):
        """
        각 알파 스텝별로 MSE를 계산하고 막대그래프로 시각화.
        - step k에서의 예측값 vs 해당 alpha에 맞는 EMA 정답
        - 최종 step에서는 원본 GT와 비교
        """
        print("\n[Alpha-step MSE Analysis] Computing per-step MSE...")

        fig_root = self._fig_root(setting)
        alpha_mse_dir = os.path.join(fig_root, 'alpha_mse')
        os.makedirs(alpha_mse_dir, exist_ok=True)

        alpha_values_desc = self._alpha_steps_desc()  # Tensor (K,) - 내림차순
        alpha_values_desc_np = alpha_values_desc.detach().cpu().numpy()
        num_steps = len(alpha_values_desc_np)

        # 각 스텝별 MSE 누적
        step_mse_sum = np.zeros(num_steps)
        step_count = 0

        model_dtype = next(self.model.parameters()).dtype
        start_mode = getattr(self.args, "use_ma_start", 0)

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Alpha MSE")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in pbar:
                batch_x, batch_y, batch_x_mark, batch_y_mark = \
                    self.process_batch_for_test(batch_x, batch_y, batch_x_mark, batch_y_mark)

                B = batch_x.shape[0]

                # EMA 정답 계산: (B, A, T_pred, C)
                ema_input = torch.cat([batch_x[:, -1:].contiguous(), batch_y], dim=1)
                ema_all, _ = self.compute_ema_sequences(ema_input)
                ema_all = ema_all[:, :, 1:].to(model_dtype)  # (B, A, T_pred, C)

                # 중간 스텝 예측값 가져오기: (B, K, T_pred, C)
                preds_all = self.sampling_with_intermediates_tensor(
                    batch_x, batch_x_mark, batch_y_mark,
                    y=batch_y, use_ma_start=start_mode,
                    return_init=False
                )

                # 각 스텝별 MSE 계산
                for k in range(num_steps):
                    pred_k = preds_all[:, k, :, :].to(self.device)  # (B, T_pred, C)

                    # 마지막 스텝 (alpha → 0)은 원본 GT와 비교
                    if k == num_steps - 1:
                        target_k = batch_y.to(self.device)  # (B, T_pred, C)
                    else:
                        # 중간 스텝은 해당 alpha의 EMA 정답과 비교
                        # alpha_values_desc[k]에 해당하는 EMA index 찾기
                        alpha_val = alpha_values_desc[k].item()
                        # self.alphas에서 가장 가까운 index
                        alpha_idx = (self.alphas - alpha_val).abs().argmin().item()
                        # 다음 스텝 (더 작은 alpha)의 EMA가 정답
                        target_alpha_idx = max(0, alpha_idx - 1)
                        target_k = ema_all[:, target_alpha_idx, :, :].to(self.device)

                    mse_k = ((pred_k - target_k) ** 2).mean().item()
                    step_mse_sum[k] += mse_k * B

                step_count += B

        # 평균 MSE 계산
        step_mse_avg = step_mse_sum / step_count

        # 로그 출력
        print("\n" + "=" * 60)
        print("[Alpha-step MSE Results]")
        print("=" * 60)
        for k in range(num_steps):
            alpha_val = alpha_values_desc_np[k]
            target_desc = "GT (original)" if k == num_steps - 1 else f"EMA(α≈{alpha_val - self.args.interval:.2f})"
            print(f"  Step {k+1:2d} (α={alpha_val:.3f}) → {target_desc}: MSE = {step_mse_avg[k]:.6f}")
        print("=" * 60 + "\n")

        # 막대그래프 저장 (wandb에는 test/mse만 로깅, 여기서는 그래프만)
        fig, ax = plt.subplots(figsize=(12, 6))

        x_labels = [f"Step {k+1}\n(α={alpha_values_desc_np[k]:.2f})" for k in range(num_steps)]
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, num_steps))  # 빨강→초록 그라데이션

        bars = ax.bar(range(num_steps), step_mse_avg, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Diffusion Step (α decreasing →)', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title(f'Per-Step MSE Analysis (Epoch {epoch})' if epoch is not None else 'Per-Step MSE Analysis', fontsize=14)
        ax.set_xticks(range(num_steps))
        ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # 각 막대 위에 값 표시
        for bar, mse_val in zip(bars, step_mse_avg):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{mse_val:.4f}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()

        # epoch별로 파일명 구분
        if epoch is not None:
            fname = f'alpha_step_mse_epoch_{epoch}.png'
        else:
            fname = 'alpha_step_mse.png'
        plt.savefig(os.path.join(alpha_mse_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved alpha-step MSE bar chart to {os.path.join(alpha_mse_dir, fname)}")

    def _visualize_all_features(self, preds, trues, test_data, test_loader, setting, epoch=None, num_samples=5):
        """
        랜덤으로 샘플을 뽑아서 전체 feature를 시각화 (과거 + 미래) - 세로 형식
        figs/<fig_tag>/<run_name>/all_features/epoch_<N>/ 에 저장
        """
        import random

        fig_root = self._fig_root(setting)

        # epoch별 폴더 생성
        if epoch is not None:
            all_feat_dir = os.path.join(fig_root, 'all_features', f'epoch_{epoch}')
        else:
            all_feat_dir = os.path.join(fig_root, 'all_features')
        os.makedirs(all_feat_dir, exist_ok=True)

        total_samples = preds.shape[0]
        num_features = preds.shape[-1]
        pred_len = preds.shape[1]
        seq_len = self.args.seq_len

        # 랜덤 샘플 선택
        sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

        # 과거 시퀀스를 다시 가져오기 위해 test_loader 순회
        inputs_list = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                inputs_list.append(batch_x.numpy())
        inputs = np.concatenate(inputs_list, axis=0)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

        for idx in sample_indices:
            input_seq = inputs[idx]  # (seq_len, num_features) - 과거
            pred = preds[idx]        # (pred_len, num_features) - 예측
            true = trues[idx]        # (pred_len, num_features) - 정답

            # 세로로 feature 나열 (1열, num_features행)
            fig, axes = plt.subplots(num_features, 1, figsize=(12, 2.5 * num_features))
            if num_features == 1:
                axes = [axes]

            fig.suptitle(f'Sample {idx} - All Features (Past + Future)', fontsize=14, y=1.02)

            for feat_idx in range(num_features):
                ax = axes[feat_idx]

                # x축 설정
                past_x = np.arange(0, seq_len)
                future_x = np.arange(seq_len, seq_len + pred_len)

                # 과거와 미래를 이어서 하나의 연속된 GT 선으로 그리기
                full_gt_x = np.concatenate([past_x, future_x])
                full_gt_y = np.concatenate([input_seq[:, feat_idx], true[:, feat_idx]])
                ax.plot(full_gt_x, full_gt_y, label='Ground Truth', color='blue', linewidth=1.5)

                # 미래 Prediction (빨간색 점선)
                ax.plot(future_x, pred[:, feat_idx], label='Prediction', color='red', linestyle='--', linewidth=1.5)

                # 과거/미래 경계선
                ax.axvline(x=seq_len - 0.5, color='black', linestyle=':', alpha=0.5, label='Forecast Start')

                ax.set_title(f'Channel {feat_idx}', fontsize=10)
                ax.set_ylabel('Value')
                if feat_idx == num_features - 1:
                    ax.set_xlabel('Time')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(all_feat_dir, f'sample_{idx}_all_features.png'), dpi=150, bbox_inches='tight')
            plt.close()

        print(f'Saved {len(sample_indices)} all-features visualization(s) to {all_feat_dir}')