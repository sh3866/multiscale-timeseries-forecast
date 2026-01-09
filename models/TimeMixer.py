import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


def get_sinusoidal_embedding(alpha, dim):
    """
    Sinusoidal embedding for alpha (diffusion style).
    alpha: (B,) → (B, dim)
    """
    device = alpha.device
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device, dtype=alpha.dtype) * -emb_scale)
    emb = alpha.unsqueeze(-1) * emb.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
    return emb

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name in ('long_term_forecast', 'short_term_forecast', 'test'):
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            # ========== MA-Diffusion Conditioning (New Architecture) ==========
            d = configs.d_model

            # 1. y_embed: y_current를 d_model 차원으로 임베딩
            if self.channel_independence == 1:
                self.y_embed = nn.Linear(1, d)  # 각 채널 독립: (B*N, P, 1) → (B*N, P, d)
            else:
                self.y_embed = nn.Linear(configs.c_out, d)  # 채널 믹싱: (B, P, C) → (B, P, d)

            # 2. Alpha embedding: sinusoidal → MLP → scale/shift
            self.alpha_embed_dim = d
            self.alpha_mlp = nn.Sequential(
                nn.Linear(d, d * 4),
                nn.GELU(),
                nn.Linear(d * 4, d * 2)  # scale, shift 출력
            )

            # 3. Fusion: concat([past_pred, y_emb]) → MLP → h
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.GELU(),
                nn.Linear(d, d)
            )

            # 4. AdaLN (LayerNorm without affine params)
            self.adaln_norm = nn.LayerNorm(d, elementwise_affine=False)

            # 5. Delta MLP: 2-layer MLP for delta prediction
            if self.channel_independence == 1:
                self.delta_mlp = nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Linear(d, 1)  # 각 채널 독립: delta (B*N, P, 1)
                )
            else:
                self.delta_mlp = nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Linear(d, configs.c_out)  # 채널 믹싱: delta (B, P, C)
                )

            # 6. Gate: alpha-dependent gating
            self.gate_layer = nn.Linear(d * 2, 1)  # scale, shift → gate scalar
            # =================================================================
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def out_projection(self, dec_out, i, out_res, ma_mode=False):
        # ma_mode는 더 이상 사용하지 않음 (새 아키텍처에서는 future_multi_mixing_ma 사용)
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y_current=None, alpha=None):
        """
        MA-Diffusion compatible forecast (New Architecture).

        Args:
            x_enc: (B, seq_len, C) - past input
            y_current: (B, pred_len, C) - current state (raw values)
            alpha: (B,) - diffusion step

        Returns:
            y_next = y_current + delta (residual prediction)
        """
        B_orig = x_enc.size(0)
        ma_mode = y_current is not None

        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # ========== MA-Diffusion: New Architecture ==========
        if ma_mode:
            # Alpha embedding: sinusoidal → MLP → scale, shift
            alpha_emb = get_sinusoidal_embedding(alpha, self.alpha_embed_dim)  # (B, d)
            alpha_cond = self.alpha_mlp(alpha_emb)  # (B, 2*d)
            alpha_scale = alpha_cond[:, :self.configs.d_model]  # (B, d)
            alpha_shift = alpha_cond[:, self.configs.d_model:]  # (B, d)

            # Gate from alpha
            gate = torch.sigmoid(self.gate_layer(alpha_cond))  # (B, 1)

            # Future prediction with MA conditioning
            dec_out_list = self.future_multi_mixing_ma(
                B, enc_out_list, x_list, y_current, alpha_scale, alpha_shift, gate
            )
        else:
            # Non-MA mode: 기존 방식 사용
            dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        # ====================================================

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        # MA mode: 직접 예측 (잔차 예측 아님)
        # 모델이 y_next를 직접 예측하도록 함

        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        """
        Future Multipredictor Mixing (Non-MA mode).
        """
        dec_out_list = []

        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # (B*N, pred_len, d_model)

                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec

                dec_out = self.projection_layer(dec_out)  # (B*N, pred_len, 1)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res, ma_mode=False)
                dec_out_list.append(dec_out)

        return dec_out_list

    def future_multi_mixing_ma(self, B, enc_out_list, x_list, y_current, alpha_scale, alpha_shift, gate):
        """
        Future Multipredictor Mixing with MA-Diffusion conditioning.

        New Architecture:
        1. past_pred = predict_layers(enc_out) → (B, pred_len, d_model)
        2. y_emb = y_embed(y_current) → (B, pred_len, d_model)
        3. h = fusion_mlp(concat([past_pred, y_emb])) → (B, pred_len, d_model)
        4. h = AdaLN(h, alpha_scale, alpha_shift) → (B, pred_len, d_model)
        5. delta = delta_mlp(h) * gate → (B, pred_len, C)
        """
        dec_out_list = []
        N = self.configs.c_out

        if self.channel_independence == 1:
            x_list_inner = x_list[0]

            # y_current: (B, pred_len, C) → (B*N, pred_len, 1) for channel independence
            y_for_embed = y_current.permute(0, 2, 1).contiguous().reshape(B * N, self.pred_len, 1)

            # Alpha conditioning 확장: (B, d) → (B*N, d)
            alpha_scale_exp = alpha_scale.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)  # (B*N, d)
            alpha_shift_exp = alpha_shift.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)  # (B*N, d)
            gate_exp = gate.unsqueeze(1).repeat(1, N, 1).reshape(B * N, 1)  # (B*N, 1)

            for i, enc_out in zip(range(len(x_list_inner)), enc_out_list):
                # 1. past_pred: enc_out → predict_layers → (B*N, pred_len, d_model)
                past_pred = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)

                if self.use_future_temporal_feature:
                    past_pred = past_pred + self.x_mark_dec

                # 2. y_emb: (B*N, pred_len, 1) → (B*N, pred_len, d_model)
                y_emb = self.y_embed(y_for_embed)

                # 3. Fusion: concat + MLP
                h = torch.cat([past_pred, y_emb], dim=-1)  # (B*N, pred_len, 2*d_model)
                h = self.fusion_mlp(h)  # (B*N, pred_len, d_model)

                # 4. AdaLN: LayerNorm → scale & shift
                h = self.adaln_norm(h)  # (B*N, pred_len, d_model)
                # alpha_scale_exp: (B*N, d) → (B*N, 1, d)
                h = h * (1 + alpha_scale_exp.unsqueeze(1)) + alpha_shift_exp.unsqueeze(1)

                # 5. Delta MLP + Gate
                delta = self.delta_mlp(h)  # (B*N, pred_len, 1)
                delta = delta * gate_exp.unsqueeze(1)  # gating

                # Reshape: (B*N, pred_len, 1) → (B, pred_len, C)
                delta = delta.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(delta)

        else:
            # Channel mixing mode
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                # 1. past_pred
                past_pred = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)

                # 2. y_emb: (B, pred_len, C) → (B, pred_len, d_model)
                y_emb = self.y_embed(y_current)

                # 3. Fusion
                h = torch.cat([past_pred, y_emb], dim=-1)
                h = self.fusion_mlp(h)

                # 4. AdaLN
                h = self.adaln_norm(h)
                h = h * (1 + alpha_scale.unsqueeze(1)) + alpha_shift.unsqueeze(1)

                # 5. Delta MLP + Gate
                delta = self.delta_mlp(h)  # (B, pred_len, C)
                delta = delta * gate.unsqueeze(1)

                dec_out_list.append(delta)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, y_current=None, alpha=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast', 'test'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, y_current=y_current, alpha=alpha)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')


class MADiffusionModelWrapper(nn.Module):
    """
    MA-Diffusion compatible wrapper for TimeMixer.

    This wrapper provides a unified interface compatible with the DiT-style
    training loop in test.py, which calls:
        model(y_current, x_past, alpha)

    The wrapper translates this to TimeMixer's interface.
    """

    def __init__(self, configs):
        super().__init__()
        self.timemixer = Model(configs)
        self.configs = configs

    def forward(self, y_current, x_past, alpha):
        """
        MA-Diffusion interface:
            y_current: (B, pred_len, C) - current state
            x_past: (B, seq_len, C) - past observations
            alpha: (B,) - diffusion step

        Returns:
            y_next: (B, pred_len, C) - y_current + delta (residual prediction)
        """
        # TimeMixer expects: x_enc, x_mark_enc, x_dec, x_mark_dec
        # For MA-Diffusion, we pass:
        #   x_enc = x_past (past observations)
        #   x_mark_enc = None (no temporal marks)
        #   x_dec = None (not used in forecast)
        #   x_mark_dec = None (not used in forecast)
        #   y_current = current state for residual
        #   alpha = diffusion step

        return self.timemixer(
            x_enc=x_past,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None,
            y_current=y_current,
            alpha=alpha
        )

    def predict_mu(self, x_past):
        """
        Predict mean of future sequence from past observations.
        Used for initialization in use_ma_start=2 mode.

        For TimeMixer, we can either:
        1. Add a dedicated mu_head (requires training)
        2. Use the model's direct prediction as mu estimate

        Here we use option 2: run TimeMixer without conditioning
        and take the mean as mu estimate.
        """
        with torch.no_grad():
            # Get direct prediction without y_current conditioning
            pred = self.timemixer(
                x_enc=x_past,
                x_mark_enc=None,
                x_dec=None,
                x_mark_dec=None,
                y_current=None,
                alpha=None
            )
            # Return mean across time dimension
            mu = pred.mean(dim=1)  # (B, C)
        return mu
