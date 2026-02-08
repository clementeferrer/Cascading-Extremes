# Second Phase: Full-Sphere Cascading Extremes with Laplace Margins

**Authors:** De Carvalho, Ferrer & Vallejos
**Dashboard:** `streamlit run app_phase2.py`

---

## Overview

Phase 2 extends the cascading extremes model from the **positive simplex** (Phase 1) to the **full unit sphere**, enabling the model to capture both market **crashes and rallies** with sign information. The mathematical reformulation replaces exponential margins with Laplace margins, Dirichlet directions with von Mises-Fisher directions, and adds formal immigration-branching genealogy with Ogata thinning simulation.

---

## What Changed: Phase 1 vs Phase 2

| Component | Phase 1 (simplex) | Phase 2 (sphere) |
|-----------|-------------------|-------------------|
| Margins | Exponential: Y = -log(1-F(z)) >= 0 | Laplace: X = F_Lap^{-1}(F(z)) in R |
| Norm | L1: R = sum(X_j) | L2: R = \|\|X\|\|_2 |
| Angular space | Positive simplex S^{d-1}_+ | Full sphere S^{d-1} |
| Direction dist | Mixture of Dirichlet | Mixture of von Mises-Fisher |
| Features | zeta(W) = log(W) - mean(log(W)) | xi(W) = [W, W^2, W_iW_j] (spherical) |
| Hawkes kernel | k*exp(-dt/tau)*phi(R) | A*exp(-dt/tau)*kappa(W_i,W_j)*phi(R) with attenuation |
| Genealogy | psi/lambda ratio only | Explicit parent P_i, cluster C(k), immigration-branching |
| Simulation | Direct exponential sampling | Ogata thinning (exact PP simulation) |
| Constraints | None | Subcriticality penalty (spectral radius < 1) |
| Token dim (d=3) | 8 = 2*3 + 2 | 14 = 3 + 9 + 2 |

---

## Module Structure

```
second_phase/
├── __init__.py          # Package init
├── CLAUDE.md            # This file
├── utils.py             # Re-exports from cascades.utils + normalize_to_sphere(), log_vmf_normalizing()
├── preprocess.py        # GARCH reuse + PIT to Laplace margins (laplace_quantile, laplace_cdf, standardize_laplace)
├── dataset.py           # L2 radial-angular decomposition, spherical_features xi(W), SphereEventSequenceDataset
├── extremes.py          # SphericalQuantileMLP: directional quantile on S^{d-1} via pinball loss
├── distributions.py     # vMF density + Wood's algorithm sampling, mixture sampling, truncated Gamma, kernel math
├── model.py             # SphericalCascadeTransformer: vMF direction + gauge magnitude + Hawkes+attenuation time
├── train.py             # Full pipeline: download -> GARCH -> Laplace PIT -> L2 -> threshold -> train
├── simulate.py          # Ogata thinning, generate_with_genealogy(), prompt_generate()
└── genealogy.py         # Parent probabilities, cluster extraction, Genealogy dataclass
```

**External files:**
- `configs/phase2.yaml` — All hyperparameters
- `app_phase2.py` — Streamlit dashboard (project root)

---

## Mathematical Details

### 1. Laplace Margins (`preprocess.py`)

Raw returns go through GARCH(1,1) filtering (reused from Phase 1), then PIT maps to standard Laplace margins:

```
X_j = F_Lap^{-1}(F_hat_j(z_j))
```

where `F_Lap^{-1}(u) = -sign(u - 0.5) * log(1 - 2|u - 0.5|)`. Values are real-valued (negative for rallies, positive for crashes).

### 2. L2 Radial-Angular Decomposition (`dataset.py`)

```
R = ||X||_2,   W = X / ||X||_2  in  S^{d-1}
```

W lives on the **full unit sphere** (not the positive simplex), so negative components encode rally directions.

Spherical features: `xi(W) = [W, W^2, W_i*W_j for i<j]` giving `d*(d+3)/2` features (9 for d=3).

Token: `[W, xi(W), log R, log dT]` of dimension 14 for d=3.

### 3. Von Mises-Fisher Direction Head (`distributions.py`, `model.py`)

The next direction is sampled from a mixture of vMF distributions:

```
W_{i+1} | H_i ~ sum_k pi_k * vMF(mu_k, kappa_k)
```

- `pi = softmax(pi_head(h))` — mixture weights
- `mu_k = normalize(mu_head_k(h))` — mean directions (unit vectors)
- `kappa_k = softplus(kappa_head(h))` — concentrations

Sampling uses Wood's (1994) rejection algorithm. For d=3, the normalizing constant has the analytic form: `log C_3(kappa) = log(kappa) - log(4*pi) - log(sinh(kappa))`.

### 4. Hawkes Intensity with Directional Attenuation (`model.py`)

```
lambda_i = mu(h_i) + sum_j k(h_i,h_j) * exp(-dt/tau) * kappa(W_i,W_j) * phi(R_j)
```

The **attenuation** `kappa(W_i, W_j) = sigmoid(MLP([W_i, W_j]))` is in (0, 1] and captures directional cross-group transmission. It never reaches zero, allowing rare cross-sector contagion.

### 5. Subcriticality Penalty (`model.py`, `train.py`)

The loss includes a penalty term enforcing the Hawkes process is subcritical:

```
Loss = -E[log p(W) + log p(R) + log p(dT)] + lambda_subcrit * max(0, max_row_sum(kernel) - margin)^2
```

This ensures finite expected cluster sizes and stable generation.

### 6. Immigration-Branching Genealogy (`genealogy.py`)

Each event has a **parent variable** P_i:
- P_i = -1: immigrant (exogenous shock)
- P_i = j: triggered by event j (endogenous cascade)

Parent probabilities: `P(P_i = immigrant) = mu_i / lambda_i`, `P(P_i = j) = kernel[i,j] / lambda_i`.

**Clusters** are extracted via BFS from immigrant roots.

### 7. Ogata Thinning Simulation (`simulate.py`)

Exact point process simulation:
1. Compute intensity bound `Lambda* = safety_factor * lambda(T_last)`
2. Propose `dt ~ Exp(Lambda*)`
3. Sample candidate mark `(W, R)` from vMF mixture + truncated Gamma
4. Accept with prob `lambda(t*, m*) / Lambda*`
5. On acceptance: append to history, sample parent
6. Stop when `T > horizon` or `max_events` reached

`prompt_generate()` creates a single shock event and runs Ogata thinning from it.

---

## Streamlit Dashboard (`app_phase2.py`)

Three modes accessible via the sidebar:

### Story Mode
- **Laplace margin histogram** — verifies symmetric, heavy-tailed, centered at 0
- **3D sphere scatter** — events on S^{d-1} colored by magnitude R, with wireframe unit sphere
- **Event timeline** — temporal distribution of extremes
- **Intensity decomposition** — mu (exogenous) + psi (endogenous) stacked
- **Cascade probability** — psi/lambda time series
- **Real vs Generated** — side-by-side sphere scatter + radial magnitude comparison

### Analyst Mode
- **Sphere scatter** — colorable by R or cascade probability
- **Intensity decomposition** — with full Hawkes kernel
- **Subcriticality diagnostic** — kernel row sums vs bound (bar chart)
- **Attenuation heatmap** — learned kappa(W_i, W_j) for sample directions
- **Hawkes kernel matrix** — full influence heatmap
- **Prompt generator** — input asset + magnitude + horizon, runs Ogata thinning, shows generated cascade on sphere + timeline + genealogy tree

### Genealogy Mode
- **Summary metrics** — event count, immigrant count, cluster count
- **Genealogy tree** — directed graph of parent-child relationships colored by cascade probability
- **Cascade probability distribution** — histogram of psi/lambda
- **Cluster explorer** — select a cluster, view its chain on the sphere with root highlighted
- **Cluster detail** — per-event breakdown with T, R, W, parent, cascade prob
- **Subcriticality check** — kernel row sums diagnostic

---

## How to Run

```bash
# Train the Phase 2 model
python3 -m second_phase.train --config configs/phase2.yaml

# Generate events via Ogata thinning
python3 -m second_phase.simulate --config configs/phase2.yaml

# Launch the dashboard
streamlit run app_phase2.py
```

Artifacts are saved to `artifacts/phase2/` (model.pt, quantile_model.pt, cdfs.npz, meta.json, simulated_events.npz, genealogy.npz).

---

## Reuse from Phase 1

| What | Source |
|------|--------|
| `download()` | `cascades.data.download` |
| `compute_log_returns()` | `cascades.preprocess` |
| `fit_garch()` | `cascades.preprocess` |
| `EmpiricalCDF` | `cascades.utils` |
| `load_config, save_json, seed_all, ensure_dir` | `cascades.utils` |
| Streamlit styling patterns | `app.py` |

---

## Verification Checklist

1. **Laplace margins**: Histogram is symmetric, heavy-tailed, centered at 0
2. **Sphere decomposition**: ||W||_2 = 1 for all events; W has negative components
3. **Threshold**: Directional variation in u_tau(W) is sensible
4. **Training**: Loss converges; subcriticality penalty stays small
5. **Generation**: Generated sphere events are statistically similar to real
6. **Genealogy**: Clusters have finite size; immigrants are a fraction of all events; cascade_prob > 0.5 for non-immigrant events

---

(c) 2024-2026 De Carvalho, Ferrer & Vallejos
