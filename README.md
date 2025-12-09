Here is the complete content for the **`README.md`** file, specifically structured for GitHub and incorporating the essential information about your library's advanced flow matching features, data loading modes, and usage instructions.

I've ensured the three data loading modes are prominently featured.

-----

# 🧬 Metagenomics Conditional Flow Matching (CFM) Library

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

A comprehensive and modular library for high-dimensional generative modeling using **Conditional Flow Matching (CFM)**. This tool is designed to handle complex datasets like those found in metagenomics, supporting various path interpolants, advanced **OT-based coupling**, and both **ODE** and **SDE** integration for sampling.

-----

## 1\. 🛠️ Setup and Dependencies

### Prerequisites

Ensure you have the required external libraries installed:

```bash
pip install numpy pandas torch scikit-learn tqdm
```

### Library Structure

The library requires a specific package structure. Ensure your repository contains the `metagenomics_cfm` directory with all the core Python files (e.g., `machine.py`, `config.py`, `coupling.py`).

-----

## 2\. 💾 Data Preparation and Loading Modes

The library loads data directly from **CSV files** via the `data_path` parameter. It automatically handles feature selection, categorical **One-Hot Encoding**, and train/validation splitting.

### Data Loading Modes

You must choose one of the following modes by setting the appropriate parameters in your configuration:

| Mode | Configuration Parameters | Description |
| :--- | :--- | :--- |
| **Mode 1: Single File + Column** | `data_path='file.csv'`, **`condition_column_name='Age_Group'`** | Features ($\mathbf{X}$) and condition ($\mathbf{Y}$) are in one CSV file. **Recommended method.** |
| **Mode 2: Two Separate Files** | `data_path='features.csv'`, **`cond_path='metadata.csv'`** | $\mathbf{X}$ and $\mathbf{Y}$ are split across two aligned CSV files. |
| **Mode 3: Unconditional** | `data_path='file.csv'`, **`condition_column_name=None`** | Training a generative model using only features ($\mathbf{X}$), ignoring any conditional columns. |

-----

## 3\. 🚀 Core Workflow: One-Shot Generation

Use the `generate_samples_from_csv` function for the fastest end-to-end workflow: loading, training (with a split), and sampling in one call.

### Conditional Example (Mode 1)

```python
from metagenomics_cfm import generate_samples_from_csv

# 1. Define input and desired output
DATA_FILE = "data/user_metrics.csv" 
NUM_SAMPLES = 50

# 2. Run the complete pipeline
synthetic_data = generate_samples_from_csv(
    data_path=DATA_FILE,
    num_samples=NUM_SAMPLES,
    
    # Mode 1 Setup
    condition_column_name="Device_Type", 
    
    # Training Overrides
    epochs=40, 
    device='cuda',
)
```

### Conditional Sampling Control

To control the generated output, you must provide the exact **one-hot template vector** ($\mathbf{y}_{template}$) that represents the desired category.

```python
# Assuming the vector [0.0, 1.0, 0.0] represents the target category ('Mobile')
template_np = np.array([[0.0, 1.0, 0.0]]) 

generated_samples = cfm_machine.sample(
    y_cond=torch.from_numpy(template_np).to(cfg.device), 
    num_samples=100
)
```

-----

## 4\. 🔬 Supported Flow Matching Variants


***

### A. Advanced Conditional Flow Matching (CFM) Variants

These flow matching strategies utilize specific endpoint couplings ($q(\mathbf{z})$) and path definitions to achieve optimal transport, stochastic flows, or geometrically specific generation.

| CFM Variant Name | Target Joint $q(\mathbf{z})$ | Path Strategy / Geometry | Implementation Key |
| :--- | :--- | :--- | :--- |
| **ConditionalFlowMatcher** | $q(\mathbf{x}_0)q(\mathbf{x}_1)$ (Independent) | Typically **Linear Path** | `"independent"` coupling |
| **VariancePreservingCFM** | $q(\mathbf{x}_0)q(\mathbf{x}_1)$ (Independent) | Conditional Gaussian path (**VP path**) | `"variance_preserving_cfm"` interpolant |
| **TargetConditionalFlowMatcher** | Implicit $\pi(\mathbf{x}_0 \mid \mathbf{x}_1)$ | Linear path from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ to data $\mathbf{x}_1$ | `"target_cfm"` coupling |
| **ExactOT-CFM / SchrodingerBridgeCFM** | $pi(x0, x1), pi_epsilon(x0, x1)$ | Linear path | `"schrodinger_bridge_cfm"` coupling (uses **Sinkhorn approx.**) |

***
### B. Path Geometries and Solvers

These define the specific nature of the interpolation $\mathbf{x}(t)$.

| Key | Geometry / Dynamics | Use Case |
| :--- | :--- | :--- |
| **`"linear"`** | Standard straight-line path. | General data, simplest flow. |
| **`"alpha_flow"`** | **Generalized Geodesic Path.** | Compositional data, flows on statistical manifolds. |
| **`"log"`** | Log-Euclidean path. | **Compositional data** (counts, relative abundances), ensuring positivity. |
| **`"sparseaware"`** | Zero-aware path. | Improves stability for **sparse data** by smoothing paths near zero. |
| **`"vp"`** / **`"variance_preserving_cfm"`** | Maintains variance structure over time. | Flows built on diffusion model principles. |

#### SDE vs. ODE Sampling

The sampling method depends on the **training method** (`flow_variant`).

| Mode | `flow_variant` (Training) | `solver` (Sampling) | Behavior |
| :--- | :--- | :--- | :--- |
| **Deterministic (ODE)** | `"deterministic"` (Default) | `"rk4"` (Default), `"euler"`, `"heun"` | Integration is exact and repeatable. |
| **Stochastic (SDE)** | **`"stochastic"`** | **`"euler_maruyama"`** (Auto-selected) | Integration includes a diffusion term, yielding diverse, noisy samples. |


## 💻 Model Architectures for Velocity Field

The choice of model architecture is controlled by the `model_type` parameter in your configuration. These models are designed to map the noisy input state ($\mathbf{x}_t$), time ($t$), and condition ($\mathbf{y}$) to the target velocity vector ($\mathbf{v}_\theta$).

| Model Key | Architecture | Description | Use Case |
| :--- | :--- | :--- | :--- |
| **`"mlp"`** | Multi-Layer Perceptron (MLP) | The standard feed-forward network baseline. Suitable for low- to moderate-dimensional, unstructured numerical data. | General tabular data, quick iteration. |
| **`"transformer"`** | Transformer Encoder | Uses a self-attention mechanism to model long-range feature dependencies, treating features as tokens. | High-dimensional data (like metagenomics) where relationships between far-apart features are critical. |
| **`"autoencoder_latent"`** | Uses an MLP, but the path geometry (`interpolant="latent"`) is defined in a learned Autoencoder space. | Generation on a learned manifold; used when the intrinsic dimensionality of the data is much lower than the observation space. | Dimensionality reduction, non-linear manifold learning. |

---

### E. Specialized Loss Functions

While the **L2 loss** (`"l2"`) on the velocity field is the default for all CFM variants, the library includes specialized losses for robustness and incorporating specific probabilistic assumptions (like those needed for Exponential Family Variational Flow Matching).

| Loss Key | Underlying Principle | Use Case |
| :--- | :--- | :--- |
| **`"l2"`** | Euclidean Distance on velocity: $E[\|v_\theta - v^*\|^2]$ | Default for all standard/OT/SB/VP flows. |
| **`"huber"`, `"l1"`, `"charbonnier"`** | Robust Regression | Use when training data or velocity fields are noisy, prone to outliers. |
| **`"poisson"`** | Poisson Negative Log-Likelihood (NLL) | Count data regression (predicting rates); suitable for exponential family assumptions. |
| **`"vfm_poisson_nll"`** | **EF-VFM Proxy Loss** | Specifically designed for **Exponential Family Variational Flow Matching** (EF-VFM) on count-like data (e.g., metagenomics). |
| **`"ot_sinkhorn"`** | Optimal Transport (OT) Distance | Used as a **metric for evaluation** or as an advanced loss for comparing predicted flow density to target density. |

---
