<div align="center">
  <h2><b> Multiscale TimeSeries Forecasting </b></h2>
</div>

## Overview

This repository implements a **Diffusion Transformer (DiT)** for long-term time series forecasting. Unlike standard diffusion models that start from random Gaussian noise, our approach leverages **multi-scale moving average** as a structured corruption process. The key insight is that exponential moving averages (EMA) with varying smoothing parameters create a natural hierarchy from heavily smoothed signals (capturing trends) to the original signal (containing fine-grained details).

## Model Architecture

### Diffusion Transformer (DiT)

The core architecture is a Diffusion Transformer that iteratively refines predictions from a smoothed state to the target signal. The model consists of the following components:

#### 1. Input Embeddings

Given an input historical sequence $\mathbf{x} \in \mathbb{R}^{T_{in} \times C}$ and a corrupted prediction sequence $\mathbf{y}^{(\alpha)} \in \mathbb{R}^{T_{pred} \times C}$, we first project them into the hidden space:

$$\mathbf{h}_x = \mathbf{W}_x \mathbf{x} + \mathbf{b}_x, \quad \mathbf{h}_y = \mathbf{W}_y \mathbf{y}^{(\alpha)} + \mathbf{b}_y$$

where $\mathbf{W}_x, \mathbf{W}_y \in \mathbb{R}^{d_{model} \times C}$ are learnable projection matrices.

#### 2. Timestep Embedding

The smoothing level $\alpha \in [0, 1]$ is encoded using sinusoidal positional embeddings:

$$\gamma_j = \exp\left(-\frac{\log(P_{max})}{d/2} \cdot j\right), \quad j = 0, 1, \ldots, \frac{d}{2}-1$$

$$\mathbf{e}_\alpha = [\cos(\alpha \cdot \gamma_0), \sin(\alpha \cdot \gamma_0), \ldots, \cos(\alpha \cdot \gamma_{d/2-1}), \sin(\alpha \cdot \gamma_{d/2-1})]$$

This embedding is then passed through an MLP:

$$\mathbf{t} = \text{MLP}(\mathbf{e}_\alpha) = \mathbf{W}_2 \cdot \text{SiLU}(\mathbf{W}_1 \mathbf{e}_\alpha + \mathbf{b}_1) + \mathbf{b}_2$$

#### 3. Rotary Position Embedding (RoPE)

We apply RoPE to encode positional information in the attention mechanism. For position $m$ and dimension $2j$:

$$\theta_j = \text{base}^{-2j/d_{head}}, \quad \text{base} = 10000$$

The rotation is applied to query and key vectors:

$$\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \odot \cos(m\theta) + \text{rotate\_half}(\mathbf{q}) \odot \sin(m\theta)$$

where $\text{rotate\_half}(\mathbf{q})$ interleaves the negated odd indices with even indices.

#### 4. DiT Block

Each DiT block consists of three sub-layers with adaptive layer normalization (AdaLN):

**Adaptive Layer Normalization:**
Given the timestep embedding $\mathbf{t}$, we compute modulation parameters:

$$[\gamma_{msa}, \beta_{msa}, \alpha_{msa}, \gamma_{mlp}, \beta_{mlp}, \alpha_{mlp}] = \text{Linear}(\text{SiLU}(\mathbf{t}))$$

**Self-Attention with AdaLN:**

$$\mathbf{h}' = \mathbf{h} + \alpha_{msa} \odot \text{SelfAttn}\left(\text{LN}(\mathbf{h}) \cdot (1 + \gamma_{msa}) + \beta_{msa}\right)$$

**Cross-Attention:**

$$\mathbf{h}'' = \mathbf{h}' + \text{CrossAttn}(\text{LN}(\mathbf{h}'), \mathbf{h}_x)$$

where queries come from the prediction sequence and keys/values come from the historical context.

**Feed-Forward Network with AdaLN:**

$$\mathbf{h}''' = \mathbf{h}'' + \alpha_{mlp} \odot \text{MLP}\left(\text{LN}(\mathbf{h}'') \cdot (1 + \gamma_{mlp}) + \beta_{mlp}\right)$$

The MLP uses GELU activation:

$$\text{MLP}(\mathbf{z}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{z})$$

#### 5. Final Layer

The output projection applies final adaptive normalization:

$$[\gamma_{out}, \beta_{out}] = \text{Linear}(\text{SiLU}(\mathbf{t}))$$

$$\hat{\mathbf{y}} = \mathbf{W}_{out}\left(\text{LN}(\mathbf{h}^{(L)}) \cdot (1 + \gamma_{out}) + \beta_{out}\right)$$

### Complete Forward Pass

$$\hat{\mathbf{y}}^{(\alpha_{k-1})} = f_\theta(\mathbf{y}^{(\alpha_k)}, \mathbf{x}, \alpha_k)$$

where $f_\theta$ represents the entire DiT model that takes a corrupted prediction at smoothing level $\alpha_k$ and produces a refined prediction at level $\alpha_{k-1}$.

## Algorithm

### Multi-Scale Moving Average Corruption

Unlike standard diffusion that uses Gaussian noise, we use **multi-scale moving average (MA)** as the corruption process. This creates a structured denoising path from heavily smoothed signals to the original signal.

#### Transition Matrix Construction

For a sequence of length $T$, we construct transition matrices $\mathbf{K}_{k} \in \mathbb{R}^{T \times T}$ for different kernel sizes $k \in \mathcal{F}(T)$, where $\mathcal{F}(T)$ denotes the set of divisors of $T$.

For kernel size $k$, the transition matrix applies sliding window averaging:

$$[\mathbf{K}_k]_{i,j} = \begin{cases} \frac{1}{k} & \text{if } |i - j| < k \text{ (within window)} \\ 0 & \text{otherwise} \end{cases}$$

The columns are then interpolated to match the sequence length, producing a square $T \times T$ matrix.

#### Alpha-Indexed Smoothing

Given an alpha schedule $\{\alpha_0, \alpha_1, \ldots, \alpha_A\}$ where $\alpha_0 = 0$ (original signal) and $\alpha_A = 1$ (maximum smoothing), we interpolate between kernel sizes:

$$\mathbf{K}_{\alpha} = (1 - w) \cdot \mathbf{K}_{k_i} + w \cdot \mathbf{K}_{k_{i+1}}$$

where $i = \lfloor \alpha \cdot (|\mathcal{F}(T)| - 1) \rfloor$ and $w$ is the fractional part.

The smoothed signal at level $\alpha$ is:

$$\mathbf{y}^{(\alpha)} = \mathbf{K}_\alpha \mathbf{y}$$

**Properties:**
- $\alpha \approx 0$: Near identity transformation (original signal)
- $\alpha \approx 1$: Strong smoothing (trend/constant approximation)

### Training Algorithm

**Algorithm 1: Training with Step-wise Trajectory Supervision**

```
Input: Training data {(x, y)}, Model f_θ, Alpha schedule {α_0, ..., α_A}
Output: Trained model parameters θ

for each epoch do
    for each batch (x, y) do
        // Step 1: Compute EMA-corrupted targets for all alpha levels
        y_full ← concat(x[:, -1:], y)                    // Prepend last observation
        {y^(α_k)}_{k=0}^{A} ← ComputeMA(y_full)          // Multi-scale MA

        // Step 2: Sample random starting alpha index
        k_start ~ Uniform(1, A)
        y_current ← y^(α_{k_start})

        // Step 3: Reverse trajectory from α_{k_start} to α_0
        L_traj ← 0
        for k = k_start down to 1 do
            y_current ← f_θ(y_current, x, α_k)           // One denoising step
            L_step ← ||y_current - y^(α_{k-1})||²        // Intermediate supervision
            L_traj ← L_traj + L_step
        end for
        L_traj ← L_traj / k_start

        // Step 4: Final prediction loss
        L_end ← ||y_current - y||²

        // Step 5: Total loss
        L ← λ_traj · L_traj + λ_end · L_end

        // Step 6: Update parameters
        θ ← θ - η∇_θL
    end for
end for
```

### Inference Algorithm

**Algorithm 2: Iterative Refinement Sampling**

```
Input: Historical sequence x, Trained model f_θ, Alpha schedule {α_1, ..., α_A}
Output: Predicted sequence ŷ

// Step 1: Initialize prediction (from last observation)
ŷ ← repeat(x[:, -1], T_pred)                    // Constant initialization

// Step 2: Reverse diffusion (from high α to low α)
for k = A down to 1 do
    ŷ ← f_θ(ŷ, x, α_k)                          // Iterative refinement
end for

return ŷ
```

**Initialization Modes:**
- **Mode 0 (Default):** Use last observation as constant: $\hat{\mathbf{y}}_0 = \mathbf{x}_{T_{in}} \cdot \mathbf{1}^T$
- **Mode 1 (Oracle):** Use EMA of ground truth (for validation analysis)
- **Mode 2 (Learned):** Use a learned statistical predictor for the mean

### Key Insights

1. **Structured Corruption:** Moving average provides a semantically meaningful corruption path - from trends to details - unlike random Gaussian noise.

2. **Mean-Aware Initialization:** Starting from the last observation provides a reasonable estimate of the signal's central tendency, which the model then refines by adding temporal variations.

3. **Step-wise Supervision:** Training with intermediate targets at each alpha level provides dense supervision, improving gradient flow and convergence.

4. **Cross-Attention Conditioning:** Historical context guides the denoising process through cross-attention, allowing the model to leverage temporal patterns from the past.

## Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hidden_dim` | Transformer hidden dimension | 32 |
| `num_heads` | Number of attention heads | 4 |
| `num_dit_block` | Number of DiT blocks | 4 |
| `mlp_ratio` | MLP hidden dim multiplier | 4.0 |
| `interval` | Alpha step size (1/steps) | 0.01 |
| `seq_len` | Input sequence length | 96 |
| `pred_len` | Prediction horizon | 96 |
| `lambda_traj` | Trajectory loss weight | 1.0 |
| `lambda_end` | Final prediction loss weight | 1.0 |

## Usage

### Training

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh1_96_96 \
    --model Ours \
    --data ETTh1 \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --seq_len 96 \
    --pred_len 96 \
    --feature_dim 7 \
    --hidden_dim 32 \
    --num_heads 4 \
    --num_dit_block 4 \
    --interval 0.01 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs 100
```

### Evaluation

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --model_id ETTh1_96_96 \
    --model Ours \
    --data ETTh1 \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --seq_len 96 \
    --pred_len 96
```

## Project Structure

```
├── models/
│   ├── Ours.py              # DiT model implementation
│   ├── TimeMixer.py         # Alternative baseline
│   └── __init__.py
├── exp/
│   ├── exp_long_term_forecasting.py   # Main training loop
│   └── global_loss.py       # Alternative training approach
├── data_provider/
│   ├── data_factory.py
│   └── data_loader.py
├── layers/
│   ├── SelfAttention_Family.py
│   ├── Embed.py
│   └── Transformer_EncDec.py
├── utils/
│   ├── tools.py
│   ├── metrics.py
│   └── losses.py
├── scripts/
│   └── main_script.sh
└── run.py                   # Entry point
```

## Mathematical Summary

**Forward Process (Corruption):**
$$\mathbf{y}^{(\alpha)} = \mathbf{K}_\alpha \mathbf{y}, \quad \alpha \in [0, 1]$$

**Reverse Process (Denoising):**
$$\mathbf{y}^{(\alpha_{k-1})} = f_\theta(\mathbf{y}^{(\alpha_k)}, \mathbf{x}, \alpha_k)$$

**Training Objective:**
$$\mathcal{L} = \lambda_{traj} \cdot \frac{1}{K} \sum_{k=1}^{K} \|\hat{\mathbf{y}}^{(\alpha_{k-1})} - \mathbf{y}^{(\alpha_{k-1})}\|^2 + \lambda_{end} \cdot \|\hat{\mathbf{y}} - \mathbf{y}\|^2$$

where $K$ is the number of denoising steps sampled during training.
