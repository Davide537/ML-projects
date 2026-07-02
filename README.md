# ML Projects

A collection of machine learning projects and coursework, ranging from classical ML fundamentals (implemented from scratch) to a research internship applying recurrent neural networks to surrogate modeling of physical (lake/storm-surge) dynamics.

## Repository Structure

| Folder | Description |
|---|---|
| [`ML basics`](./ML%20basics) | Coursework notebooks ("Linear Algebra and Optimization for Machine Learning"): logistic regression from scratch, gradient-based optimization, and a Poisson-solver comparison. |
| [`Internship Deltares`](./Internship%20Deltares) | Research internship code (Julia): LSTM and FRNN surrogate models trained to emulate a 0D lumped lake model, including storm-surge generalization and stability testing with the aim of developing accurate and fast weather forecasting. |

## Internship Deltares

Julia-based project (using [Flux.jl](https://fluxml.ai/)) that trains recurrent neural network surrogates to emulate a physics-based **0D lumped lake model**, then evaluates how well they generalize and unroll stably over time, including under storm conditions.

**Contents**
- `training_LSTM.ipynb` — LSTM model: time-series cross-validation (TSCV), model architecture, training, and multi-step unrolling; includes a worked example plus experiments comparing model alternatives on storm data.
- `training_FRNN.ipynb` — Fully-Recurrent Neural Network (FRNN) model: architecture, training, unrolling, a worked example, and dedicated stability/generality tests (unroll stability across trials, performance on held-out storm datasets).
- `Dataset/model_0d_lake.ipynb` — defines the lumped lake model (governing ODEs for water level `h₀`, surge height `h_s`, and current `u_c`) and generates the normal and storm training datasets, saved as `.jld2` files with accompanying plots.
- `Make plots.ipynb` — generates the final LSTM and FRNN report plots from saved training/unrolling data.
- `Dense plots/`, `LSTM plots/` — figures and saved intermediate data (loss curves, unrolling comparisons, stability plots) used in the project report.
- `Project.toml` / `Manifest.toml` — Julia environment/dependency specification (Flux, FluxTraining, CUDA/cuDNN, OrdinaryDiffEq, JLD2, Zygote, MLUtils, HyperTuning, etc.).

See [`Internship Deltares/README.md`](./Internship%20Deltares/README.md) for the author's original notes on file usage.

## ML Basics

Coursework notebooks for **Linear Algebra and Optimization for Machine Learning**:

- `LAOML_P1.ipynb` — binary classification of malicious Android applications from permission data: data loading and sparsity analysis, one-hot encoding, train/test splitting, and logistic regression implemented from scratch (cost function, gradient derivation, and gradient descent).
- `LAOML_P2.ipynb` — a Poisson-equation solver and comparison of solution methods.
