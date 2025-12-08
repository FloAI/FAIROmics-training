# 🧬 Metagenomics Conditional Flow Matching (CFM) Library

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

A comprehensive and modular library for high-dimensional generative modeling using **Conditional Flow Matching (CFM)**. This tool is designed to handle complex datasets like those found in metagenomics, supporting various path interpolants, optimal transport coupling, and both **ODE** and **SDE** integration for sampling.

-----

## 1\. 🛠️ Setup and Dependencies

### Prerequisites

Ensure you have the required external libraries installed:

```bash
pip install numpy pandas torch scikit-learn tqdm
```

### Library Structure

Ensure your project repository contains the `metagenomics_cfm` directory with all necessary files, as the library relies on precise relative imports.

-----

## 2\. 💾 Data Preparation and Modes

The library loads data directly from **CSV files** via the `data_path` parameter. It automatically handles feature selection, categorical **One-Hot Encoding**, and train/validation splitting.

### Data Loading Modes

| Mode | Configuration | Description |
| :--- | :--- | :--- |
| **Mode 1: Single File + Column** | `data_path='file.csv'`, **`condition_column_name='Age_Group'`** | Features ($\mathbf{X}$) and condition ($\mathbf{Y}$) are in one CSV file. **Recommended method.** |
| **Mode 2: Two Separate Files** | `data_path='features.csv'`, **`cond_path='metadata.csv'`** | $\mathbf{X}$ and $\mathbf{Y}$ are split across two aligned CSV files. |

-----

## 3\. 🚀 Core Workflow: One-Shot Generation

Use the `generate_samples_from_csv` function for the fastest end-to-end workflow: loading, training (with a split), and sampling in one call.

### Conditional Example

```python
from metagenomics_cfm import generate_samples_from_csv

# 1. Define input and desired output
DATA_FILE = "data/user_metrics.csv" 
NUM_SAMPLES = 50

# 2. Run the complete pipeline
synthetic_data = generate_samples_from_csv(
    data_path=DATA_FILE,
    num_samples=NUM_SAMPLES,
    
    # Specify the column to condition on (Mode 1 setup)
    condition_column_name="Device_Type", 
    
    # Optional Overrides:
    epochs=40, 
    device='cuda',
)

print(f"Generated data shape: {synthetic_data.shape}")
```

### Conditional Sampling Control

To control the generated output, you must provide the exact **one-hot template vector** ($\mathbf{y}_{template}$) that represents the desired category.

The loader automatically performs **One-Hot Encoding** on categorical columns:

The template vector must be shaped $(1, C)$ and match the dimensions created by the automatic encoding of your categorical column.

```python
# The vector must be manually verified to match the encoding order (e.g., [Desktop, Mobile, Tablet])
template_np = np.array([[0.0, 1.0, 0.0]]) 

generated_samples = cfm_machine.sample(
    y_cond=torch.from_numpy(template_np).to(cfg.device), 
    num_samples=100
)
```

-----

## 4\. 🔬 Advanced CFM Features and Dynamics

### A. Path Interpolants (`interpolant`)

The CFM objective minimizes the difference between the model's velocity field ($\mathbf{v}_\theta$) and the true path velocity ($\mathbf{v}^*$) defined by the interpolant between $\mathbf{x}_0$ (data) and $\mathbf{x}_1$ (prior).

| Key | Description | Best Use Case |
| :--- | :--- | :--- |
| **`"linear"`** | Standard straight-line interpolation (Euclidean). | General data, baseline testing. |
| **`"log"`** | Log-space interpolation. | **Compositional data** (CLR-transformed counts) where values must remain positive/relative. |
| **`"spherical"`** | Interpolation on a hypersphere. | Normalized data, directional features. |

### B. SDE vs. ODE Solvers

The sampling method depends on the **training method** (`flow_variant`). The library handles the automatic switch to the SDE solver when noise ($\sigma > 0$) is detected in training.

| Mode | `flow_variant` (Training) | `solver` (Sampling) | Behavior |
| :--- | :--- | :--- | :--- |
| **Deterministic (ODE)** | `"deterministic"` (Default) | `"rk4"` (Default), `"euler"`, `"heun"` | Integration is exact and repeatable. |
| **Stochastic (SDE)** | **`"stochastic"`** | **`"euler_maruyama"`** (Auto-selected) | Integration includes a diffusion term, yielding diverse, noisy paths. |

#### Example SDE Configuration

```python
cfg = CFMConfig(
    # Training: Add noise to the target velocity (Stochastic Flow Matching)
    flow_variant="stochastic",         
    stochastic_noise_scale=0.2,        # Set diffusion magnitude (Sigma)
    
    # Sampling: Use the SDE integrator (set to euler_maruyama)
    solver="euler_maruyama"            
)
```

# ------------------------------------------------------------------
# Applying CLR transform
# ------------------------------------------------------------------

def generate_samples_from_csv(
    data_path: str,
    # ... (other arguments) ...
) -> np.ndarray:
    
    # ... (Step 1 & 2: Configuration and Loading) ...
    
    print(f"Loading data from {cfg.data_path}...")
    data, cond, x_dim, cond_dim = load_data_from_config(cfg)
    
    # --- START CLR INTEGRATION ---
    print("Applying Centered Log-Ratio (CLR) transformation...")
    
    # NOTE: This assumes your CSV features (data) are non-negative counts or 
    # relative abundances that require CLR. If the data is already transformed 
    # (e.g., already log-normalized), skip this step.
    
    from .data_utils import apply_clr_transform # Import CLR utility locally
    
    data = apply_clr_transform(data) 
    
    print(f"CLR successful. Feature dimension remains ({data.shape[1]}).")
    # --- END CLR INTEGRATION ---
    
    # 3. Prepare Datasets (Splitting and Machine Initialization follow here)
    cfg.cond_dim = cond_dim 
    
    # ... (rest of the function continues) ...
