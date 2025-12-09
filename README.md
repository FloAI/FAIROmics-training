That's great\! You want this comprehensive list of supported flow strategies included in your final `README.md` to highlight the library's advanced capabilities beyond standard CFM.

Here is the complete content for the **`README.md`** file, now incorporating this detailed feature list and the strategic use of image tags.

-----

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

Mode,Configuration,Description
Mode 1: Single File + Column,"data_path='file.csv', condition_column_name='Age_Group'",Features (X) and condition (Y) are in one CSV file. Recommended method.
Mode 2: Two Separate Files,"data_path='features.csv', cond_path='metadata.csv'",X and Y are split across two aligned CSV files.
Mode 3: Unconditional,"data_path='file.csv', condition_column_name=None","Training a generative model using only features (X), ignoring any conditional columns."

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
    condition_column_name="Device_Type", 
    epochs=40
)
```

### Conditional Sampling Control

To control the generated output, you must provide the exact **one-hot template vector** ($\mathbf{y}_{template}$) that represents the desired category.

The vector must be shaped $(1, C)$ and match the dimensions created by the automatic encoding of your categorical column.

```python
# The vector must be manually verified to match the encoding order (e.g., [Healthy, Diseased, Other])
template_np = np.array([[0.0, 1.0, 0.0]]) 

generated_samples = cfm_machine.sample(
    y_cond=torch.from_numpy(template_np).to(cfg.device), 
    num_samples=100
)
```

-----

## 4\. 🔬 Supported Flow Matching Variants

Your library is equipped to handle a wide range of state-of-the-art flow matching strategies, defined by their target joint distribution $q(\mathbf{z})$ and path geometry $\mathbf{x}(t)$.

### A. Endpoint Coupling Strategies ($q(\mathbf{z})$)

These variants define how the source ($\mathbf{x}_0$) and target ($\mathbf{x}_1$) endpoints are coupled, affecting the target velocity field $\mathbf{v}^*$.

| CFM Variant Name | Target Joint $q(\mathbf{z})$ | Internal Coupling Key | Description |
| :--- | :--- | :--- | :--- |
| **ConditionalFlowMatcher** | $q(\mathbf{x}_0)q(\mathbf{x}_1)$ | `"independent"` | Standard independent endpoint CFM. |
| **TargetConditionalFlowMatcher** | $q(\mathbf{x}_1)$ | `"target_cfm"` | Learns flow from Gaussian ($\mathbf{x}_0$) to Data ($\mathbf{x}_1$). |
| **VariancePreservingCFM** | $q(\mathbf{x}_0)q(\mathbf{x}_1)$ | `"independent"` | Uses independent coupling with the specialized VP path. |
| **ExactOT-CFM / SchrodingerBridgeCFM** | $\pi(\mathbf{x}_0, \mathbf{x}_1)$ or $\pi_{\epsilon}(\mathbf{x}_0, \mathbf{x}_1)$ | `"schrodinger_bridge_cfm"` | Approximates Optimal Transport (OT) or Schrödinger Bridge coupling via **minibatch Sinkhorn optimization**. |

### B. Path Geometries and Specialized Flows

These interpolation methods define the path $\mathbf{x}(t)$ regardless of endpoint coupling.

| Interpolant Key | Geometry / Concept | Primary Use Case |
| :--- | :--- | :--- |
| **`"linear"`** | Straight-line path (Standard) | General, Euclidean data, baseline testing. |
| **`"alpha_flow"`** | **Generalized Geodesic Path.** | Compositional data, flows on statistical manifolds (e.g., $\alpha=0$ is log-Euclidean). |
| **`"log"`** | Log-Euclidean path. | **Compositional data** (counts, relative abundances), ensuring positivity. |
| **`"spherical"`** | Geodesic path on a hypersphere. | Normalized data, directional features. |
| **`"vp"`** | Variance-Preserving (VP) path. | Flows where maintaining noise variance structure is key. |
| **`"sparseaware"`** | Zero-aware path. | Improves stability for sparse data by smoothing paths near zero values. |
| **`"latent"`** | Path in a lower-dimensional latent space. | Reducing interpolation complexity for high-dimensional data. |

### C. SDE vs. ODE Solvers

The sampling method depends on the **training method** (`flow_variant`).

| Mode | `flow_variant` (Training) | `solver` (Sampling) | Behavior |
| :--- | :--- | :--- | :--- |
| **Deterministic (ODE)** | `"deterministic"` (Default) | `"rk4"` (Default), `"euler"`, `"heun"` | Integration is exact and repeatable. |
| **Stochastic (SDE)** | **`"stochastic"`** | **`"euler_maruyama"`** (Auto-selected) | Integration includes a diffusion term, yielding diverse, noisy samples. |
