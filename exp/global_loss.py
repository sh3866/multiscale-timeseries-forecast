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
        
        step_losses = []
        global_losses = []
        
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
            for i, (batch_x_raw, batch_y, batch_x_mark_raw, batch_y_mark_raw) in pbar:
                
                # === step loss ===
                batch_x, batch_x_mark, batch_y_mark, batch_ema_y_prev, batch_ema_y_target, alpha_values = self.process_batch(batch_x_raw, batch_y, batch_x_mark_raw, batch_y_mark_raw)
                
                step_outputs = self.model(batch_ema_y_prev, batch_x, alpha_values)

                step_loss = criterion(step_outputs, batch_ema_y_target)
                step_losses.append(step_loss.item())
                
                # === global loss (final + intermediate EMA supervision) ===
                preds_all = self.sampling_with_intermediates_tensor(
                    batch_x_raw.float(),
                    batch_x_mark_raw.float() if batch_x_mark_raw is not None else None,
                    batch_y_mark_raw.float() if batch_y_mark_raw is not None else None,
                )  # (B, A-1, T_pred, C)

                ema_full, _ = self.compute_ema_sequences(
                    torch.cat([batch_x_raw.float()[:, -1].unsqueeze(1), batch_y.float()], dim=1),
                    interval=self.args.interval,
                )  # (B, A, T_pred+1, C)
                ema_full = ema_full[:, :, 1:]  # (B, A, T_pred, C)
                targets_all = torch.flip(ema_full, dims=[1])[:, 1:, :, :].to(self.device)  # (B, A-1, T_pred, C), [0.95..0.0]

                w_final = getattr(self.args, 'global_final_weight', 1.0)
                w_inter = getattr(self.args, 'global_intermediate_weight', 1.0)
                global_loss_final = criterion(preds_all[:, -1, :, :], batch_y.float().to(self.device))
                global_loss_intermediate = criterion(preds_all, targets_all)
                global_loss = w_final * global_loss_final + w_inter * global_loss_intermediate
                global_losses.append(global_loss.item())
                
                # === tqdm 진행바에 step+global 동시 표시 ===
                pbar.set_postfix(
                    step_loss=step_loss.item(),
                    global_loss=global_loss.item()
                )
        
        step_loss_avg = np.mean(step_losses)
        global_loss_avg = np.mean(global_losses)
        total_loss_avg = step_loss_avg + global_loss_avg

        self.model.train()
        return step_loss_avg, global_loss_avg, total_loss_avg

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

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
        #                                     steps_per_epoch=train_steps,
        #                                     pct_start=self.args.pct_start,
        #                                     epochs=self.args.train_epochs,
        #                                     max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            self.model.train()

            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            
            for i, (batch_x_raw, batch_y, batch_x_mark_raw, batch_y_mark_raw) in pbar:
                model_optim.zero_grad()
                
                batch_x, batch_x_mark, batch_y_mark, batch_ema_y_prev, batch_ema_y_target, alpha_values = self.process_batch(batch_x_raw, batch_y, batch_x_mark_raw, batch_y_mark_raw)
                
                # === 한 step loss ===
                step_outputs = self.model(batch_ema_y_prev, batch_x, alpha_values)
                
                step_loss = criterion(step_outputs, batch_ema_y_target)
                
                # === global loss (final + intermediate EMA supervision) ===
                # 1) differentiable intermediate predictions
                preds_all = self.sampling_with_intermediates_tensor(
                    batch_x_raw.float(),
                    batch_x_mark_raw.float() if batch_x_mark_raw is not None else None,
                    batch_y_mark_raw.float() if batch_y_mark_raw is not None else None,
                )  # (B, A-1, T_pred, C)

                # 2) build EMA targets for all alpha steps (reverse order)
                ema_full, _ = self.compute_ema_sequences(
                    torch.cat([batch_x_raw.float()[:, -1].unsqueeze(1), batch_y.float()], dim=1),
                    interval=self.args.interval,
                )  # (B, A, T_pred+1, C)
                ema_full = ema_full[:, :, 1:]  # drop t=0 to keep only future horizon: (B, A, T_pred, C)
                targets_all = torch.flip(ema_full[:, 1:, :, :].to(self.device), dims=[1])  # align with [1.0, ..., interval]

                # 3) final y_true term + intermediate EMA term
                t_true_y = batch_y.float().to(self.device)
                t_pred_final = preds_all[:, -1, :, :]
                global_loss_final = criterion(t_pred_final, t_true_y)
                global_loss_intermediate = criterion(preds_all, targets_all)
                global_loss = global_loss_final + global_loss_intermediate
                
                loss = global_loss + step_loss

                loss.backward()
                model_optim.step()

                # if self.args.lradj == 'TST':
                    # adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    # scheduler.step()
                    
                # === 시각화: 100 스텝마다 1개 샘플 저장 + Metric 계산 ===
                if i % 100 == 0:
                    with torch.no_grad():
                        # sampling을 이용해 원본 future 예측
                        pred_y = self.sampling(batch_x[:1], batch_x_mark[:1], batch_y_mark[:1])
                        true_y = batch_y[:1]

                        # numpy 변환
                        pred_np = pred_y.detach().cpu().numpy()
                        true_np = true_y.detach().cpu().numpy()
                        x_np = batch_x[:1].detach().cpu().numpy()

                        # 필요하면 inverse_transform 적용 (데이터셋에 따라)
                        if train_data.scale and self.args.inverse:
                            pred_np = train_data.inverse_transform(pred_np.reshape(-1, pred_np.shape[-1])).reshape(pred_np.shape)
                            true_np = train_data.inverse_transform(true_np.reshape(-1, true_np.shape[-1])).reshape(true_np.shape)
                            x_np = train_data.inverse_transform(x_np.reshape(-1, x_np.shape[-1])).reshape(x_np.shape)
                            
                        mse = np.mean((pred_np - true_np) ** 2)
                        mae = np.mean(np.abs(pred_np - true_np))
                        
                        wandb.log({
                            "epoch": epoch,
                            "iteration": i,
                            "train/sampling_mse": mse,
                            "train/sampling_mae": mae,
                        })

                        # (과거 구간 + 미래 구간)으로 concat 해서 그림 그리기
                        gt = np.concatenate((x_np[0, :, -1], true_np[0, :, -1]), axis=0)
                        pd = np.concatenate((x_np[0, :, -1], pred_np[0, :, -1]), axis=0)

                        # 그림 저장
                        if epoch is not None:
                            visual(gt, pd, os.path.join(train_fig_dir, f"epoch_{epoch}-{i}.pdf"))
                        else:
                            visual(gt, pd, os.path.join(train_fig_dir, f"{i}.pdf"))
                            
                            
                        # === step-by-step sampling =================================================
                        # 추가로 매 스텝마다 loss, MSE, MAE 출력해보기
                        preds = self.sampling_with_intermediates(batch_x[:1], batch_x_mark[:1], batch_y_mark[:1])
                        true_y = batch_y[:1].detach().cpu().numpy()

                        step_mses, step_maes = [], []
                        for step_idx, pred in enumerate(preds):
                            # shape: (1, T_pred, C)
                            if train_data.scale and self.args.inverse:
                                pred = train_data.inverse_transform(
                                    pred.reshape(-1, pred.shape[-1])
                                ).reshape(pred.shape)
                                true_np = train_data.inverse_transform(
                                    true_y.reshape(-1, true_y.shape[-1])
                                ).reshape(true_y.shape)
                            else:
                                true_np = true_y

                            mse = np.mean((pred - true_np) ** 2)
                            mae = np.mean(np.abs(pred - true_np))
                            step_mses.append(mse)
                            step_maes.append(mae)

                        # === wandb log ===
                        wandb.log({
                            "epoch": epoch,
                            "iteration": i,
                            "train/sampling_step_mse": step_mses,
                            "train/sampling_step_mae": step_maes,
                        })

                        # === 그래프 저장 ===
                        plt.figure()
                        plt.plot(step_mses, label="MSE")
                        plt.plot(step_maes, label="MAE")
                        plt.xlabel("alpha step (reverse)")
                        plt.ylabel("Loss")
                        plt.legend()
                        plt.title(f"Sampling losses at epoch {epoch}, iter {i}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(train_fig_dir, f"epoch_{epoch}-{i}_loss_curve.pdf"))
                        plt.close()
                        
                        # ============================================================================


                wandb.log({
                    "epoch": epoch,
                    "iteration": i,
                    "train/step_loss": step_loss.item(),
                    "train/global_loss": global_loss.item(),
                    "train/total_loss": loss.item()
                })

                pbar.set_postfix(loss=loss.item())

            vali_step_loss, vali_global_loss, vali_total_loss = self.vali(vali_data, vali_loader, criterion)
            test_step_loss, test_global_loss, test_total_loss = self.vali(test_data, test_loader, criterion)
            self.test(setting, test=False, epoch=epoch)

            wandb.log({
                "epoch": epoch,
                "val/loss": vali_total_loss,
                "test/loss": test_total_loss,
                "test/step_loss": test_step_loss,
                "test/global_loss": test_global_loss,
            })

            print("Epoch: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(epoch + 1, vali_total_loss, test_total_loss))
            early_stopping(vali_total_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # if self.args.lradj != 'TST':
            #     adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            # else:
            #     print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
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
