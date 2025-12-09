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

Your library supports a broad range of state-of-the-art CFM variants, defined by their coupling and path geometry.

### A. Endpoint Coupling Strategies ($q(\mathbf{z})$)

These variants define the target joint distribution of the endpoints.

| CFM Variant Name | Target Joint $q(\mathbf{z})$ | Implementation Strategy |
| :--- | :--- | :--- |
| **ConditionalFlowMatcher** | $q(\mathbf{x}_0)q(\mathbf{x}_1)$ | Standard **Independent** coupling. |
| **TargetConditionalFlowMatcher** | $q(\mathbf{x}_1)$ | Flow learned from **Gaussian Prior** ($\mathbf{x}_0$) to **Data** ($\mathbf{x}_1$). |
| **ExactOT-CFM / SchrodingerBridgeCFM** | $\pi(\mathbf{x}_0, \mathbf{x}_1)$ or $\pi_{\epsilon}(\mathbf{x}_0, \mathbf{x}_1)$ | Approximates OT/SB joint distribution via **minibatch Sinkhorn optimization**. |

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
