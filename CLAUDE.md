# Cascading Extremes: Mathematical Model and Implementation

**Authors:** De Carvalho, Ferrer & Vallejos
**Live Paper:** A generative model for cascading multivariate extremes

---

## Overview

This repository implements a **causal transformer** for modeling and generating **cascading extremes** in multivariate time series. The model combines:

1. **Extreme Value Theory (EVT)** for rigorous treatment of tail events
2. **Marked Point Processes** for temporal dynamics
3. **Hawkes Self-Excitation** for cascade propagation
4. **Transformer Architecture** for learning complex history-dependent patterns

The system learns from historical extreme events and generates realistic synthetic cascades that preserve the statistical and geometric properties of real market crashes and rallies.

---

## 1. Data Representation

### 1.1 Standardization to Laplace Margins

Raw asset returns are transformed to have **standard Laplace margins**:

1. Fit GARCH(1,1) with Student-t innovations to each asset's log-returns
2. Compute standardized residuals: `z_t^j = (r_t^j - mu_t^j) / sigma_t^j`
3. Apply the Probability Integral Transform (PIT) to Laplace margins:

```
X_t^j = F_Lap^{-1}(F_hat_j(z_t^j))
```

where `F_Lap^{-1}(u) = -sign(u - 0.5) * log(1 - 2|u - 0.5|)`.

4. Normalize by `log(n/2)` where `n` is the number of observations.

This yields `X_t = (X_t^1, ..., X_t^d)` with standard Laplace margins on the full real line (both positive and negative values).

**Implementation:** See `cascades/preprocess.py` for `standardize()`.

### 1.2 Radial-Angular Decomposition (L2 Norm)

Each observation is decomposed into:

- **Radial component** (magnitude): `R_t = ||X_t||_2` (L2/Euclidean norm)
- **Angular component** (direction): `W_t = X_t / R_t` on the unit sphere `S^{d-1}`

The direction `W` lives on the full unit sphere (not the positive simplex), capturing both crashes AND rallies with sign information.

**Implementation:** See `cascades/dataset.py` for `compute_radial_angular()`.

### 1.3 Sign-Aware Geometric Features

The centered log-ratio coordinates are computed with sign awareness:

```
zeta(W) = sign(W) * log(|W| + eps) - mean(sign(W) * log(|W| + eps))
```

This handles negative components of W that arise from the sphere representation.

**Implementation:** See `cascades/dataset.py` for `zeta()`.

---

## 2. Extreme Events as a Marked Point Process

### 2.1 Directional Threshold

An extreme event occurs when:

```
R_t > u_tau(W_t)
```

where `u_tau(w)` is a **direction-dependent quantile threshold** learned via conditional quantile regression. This captures that extremes in different asset combinations may require different magnitude thresholds.

**Implementation:** The `DirectionalQuantileMLP` in `cascades/extremes.py` learns `u_tau(w)`.

### 2.2 Event Sequence

The sequence of extreme events forms a marked point process:

```
N = sum_i delta_{(T_i, M_i)}
```

where:
- `T_i` = time of event i
- `M_i = (W_{T_i}, R_{T_i})` = mark (direction and magnitude)

---

## 3. Two-Phase Architecture

### Phase 1: Simplex Model (`cascades/`)

The original model operating on the positive simplex with exponential margins, L1 norm, and mixture of Dirichlet directions. See `cascades/model.py`.

### Phase 2: Sphere Model (`second_phase/`)

The full-sphere model with Laplace margins, L2 norm, and von Mises-Fisher directions. This is the model used for generation in the immersive viewer.

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Margins | Exponential: Y = -log(1-F(z)) >= 0 | Laplace: X = F_Lap^{-1}(F(z)) in R |
| Norm | L1: R = sum X_j | L2: R = \|\|X\|\|_2 |
| Angular space | Positive simplex S^{d-1}_+ | Full sphere S^{d-1} |
| Direction dist | Mixture of Dirichlet | Mixture of von Mises-Fisher |
| Features | zeta(W) = log(W) - mean(log(W)) | xi(W) = [W^2, W_iW_j] (spherical) |
| Hawkes kernel | k*exp(-dt/tau)*phi(R) | A*K(dt)*kappa(m,m')*phi(R) with attenuation |
| Genealogy | psi/lambda ratio only | Explicit parent P_i, cluster C(k) |
| Simulation | Direct exponential sampling | Ogata thinning (exact PP simulation) |
| Constraints | None | Subcriticality penalty (spectral radius < 1) |

---

## 4. Phase 2 Generative Model (Active)

### 4.1 Direction Head (Mixture of von Mises-Fisher)

The next direction is sampled from a **mixture of vMF distributions** on the sphere:

```
W_{i+1} | H_i ~ sum_k pi_{i,k} * vMF(mu_{i,k}, kappa_{i,k})
```

where:
- `pi_i = softmax(Pi(h_i))` = mixture weights
- `mu_{i,k} = normalize(M_k(h_i))` = mean direction on sphere
- `kappa_{i,k} = softplus(K(h_i))` = concentration parameter

The vMF density on S^{d-1} for d=3: `C_3(kappa) = kappa / (4*pi*sinh(kappa))`.

**Implementation:** See `second_phase/model.py` and `second_phase/distributions.py`.

### 4.2 Magnitude Head (Truncated Gamma)

Same as Phase 1: `R_{i+1} | {W_{i+1}, R > u} ~ TruncGamma(a, beta_{i+1})`.

### 4.3 Time Head (Hawkes with Directional Attenuation)

```
lambda_i = mu(h_i) + sum_{j<=i} A * exp(-dt/tau) * kappa(W_i, W_j) * phi(R_j)
```

The attenuation `kappa(W_i, W_j) = sigmoid(MLP([W_i, W_j]))` in (0,1] allows directional cross-group transmission.

A **subcriticality penalty** `max(0, max_row_sum(kernel) - margin)^2` ensures the process doesn't explode.

### 4.4 Immigration-Branching Genealogy

Each event has an explicit parent variable:
- `P(P_i = 0) = mu_i / lambda_i` (immigrant — exogenous shock)
- `P(P_i = j) = kernel[i,j] / lambda_i` (offspring of event j)

Clusters are extracted via BFS from immigrant events.

**Implementation:** See `second_phase/genealogy.py`.

### 4.5 Ogata Thinning Simulation

Generation uses exact point process simulation:
1. Compute intensity bound `Lambda* = safety_factor * Lambda(T_last)`
2. Propose `dt ~ Exp(Lambda*)`
3. Sample candidate mark `(W, R)` from vMF mixture + truncGamma
4. Accept with probability `lambda(t*, m*) / Lambda*`
5. Stop when `T > horizon` or `max_events` reached

**Implementation:** See `second_phase/simulate.py`.

---

## 5. Cascade Probability

A key interpretable quantity is the **cascade probability**:

```
P(cascade | event i) = psi_i / lambda_i
```

where:
- `psi_i = sum_j kappa(.) * phi(R_j)` = endogenous (self-excited) component
- `lambda_i = mu_i + psi_i` = total intensity

When `psi/lambda` is high, the event was likely triggered by previous events (a cascade). When low, it's an exogenous shock.

**Visualization:** The immersive viewer colors points by cascade probability (viridis scale: blue=exogenous, yellow=cascade).

---

## 6. Training Objective

The model is trained by maximizing the joint log-likelihood with a subcriticality penalty:

```
L = sum_i [ log p(W|h) + log p(R|h) + log p(dT|h) ] - subcrit_weight * subcrit_penalty
```

Each component:
- **Direction:** Log-likelihood under the vMF mixture
- **Magnitude:** Truncated Gamma log-density
- **Time:** Point process: `log lambda - lambda * dT`
- **Subcriticality:** `max(0, max_row_sum(kernel) - 0.95)^2`

**Implementation:** See `second_phase/train.py`.

---

## 7. Immersive 3D Viewer

The web viewer displays events on the **unit sphere** inside a `[-1.5, 1.5]^3` cube:

- Points are mapped from sphere coordinates `W` with radial scaling by `R/u_tau`
- A wireframe unit sphere shows the geometric reference
- Axes helpers show the three asset directions
- Points colored by cascade probability (viridis: blue=exogenous, yellow=cascade)
- Camera orbits around origin with zoom controls

Generation in the viewer uses the Phase 2 model (vMF + Ogata thinning) when `artifacts/phase2/model.pt` exists, with a random-segment fallback otherwise.

**Implementation:** See `web/immersive/src/scenes/` and `web/api/main.py`.

---

## 8. Project Structure

```
cascades/
├── dataset.py        # Data loading, tokenization, L2 radial-angular, sign-aware zeta
├── extremes.py       # Directional quantile model for thresholds
├── model.py          # CascadingTransformer (Phase 1, Dirichlet)
├── preprocess.py     # GARCH + PIT → Laplace margins, log(n/2) normalization
├── train.py          # Phase 1 training loop
├── simulate.py       # Phase 1 autoregressive generation
├── viz_export/       # Export runs to parquet for the viewer
└── utils.py          # Configuration, seeding, I/O utilities

second_phase/
├── dataset.py        # L2 radial-angular, spherical features, tokens (14-dim for d=3)
├── extremes.py       # SphericalQuantileMLP for sphere threshold
├── distributions.py  # vMF density/sampling (Wood 1994), truncGamma, kernels
├── model.py          # SphericalCascadeTransformer (vMF + attenuation + subcriticality)
├── train.py          # Full pipeline with subcriticality penalty
├── simulate.py       # Ogata thinning + parent sampling
├── genealogy.py      # Parent variables, cluster extraction, Genealogy dataclass
├── preprocess.py     # GARCH + PIT → Laplace margins (imports from cascades)
└── utils.py          # Re-exports + sphere helpers (log_vmf_normalizing)

web/
├── api/
│   ├── main.py       # FastAPI backend (Phase 2 generation via Ogata thinning)
│   ├── storage.py    # Run storage (parquet I/O)
│   └── metrics.py    # Summary metrics
└── immersive/        # React + Three.js visualization
    └── src/
        ├── pages/ViewerPage.tsx      # Main viewer component
        ├── scenes/CascadeScene.tsx   # 3D scene with sphere + cube
        ├── scenes/SimplexPlane.tsx   # Wireframe unit sphere overlay
        ├── scenes/CubeFrame.tsx      # [-1.5, 1.5]^3 wireframe cube
        └── scenes/EventsPoints.tsx   # Point cloud colored by psi/lambda

configs/
├── default.yaml      # Phase 1 config
└── phase2.yaml       # Phase 2 config (vMF, attenuation, subcriticality)

app.py                # Phase 1 Streamlit dashboard
app_phase2.py         # Phase 2 Streamlit dashboard (Story, Analyst, Genealogy)
```

---

## 9. Key Equations Summary

| Component | Distribution | Parameters from Transformer |
|-----------|-------------|----------------------------|
| Direction W | Mixture of vMF | pi (weights), mu (mean dirs), kappa (concentrations) |
| Magnitude R | Truncated Gamma | a (shape), beta (rate from gauge network) |
| Time dT | Exponential (Hawkes) | lambda (intensity with attenuation) |
| Cascade Prob | psi/lambda | Ratio of endogenous to total intensity |
| Subcriticality | Penalty | max(0, max_row_sum - margin)^2 |

---

## 10. Deployment

The immersive viewer is deployed on Hugging Face Spaces via Docker:

```bash
git push space main
```

The Dockerfile builds the React frontend, copies `cascades/`, `second_phase/`, `configs/`, and `artifacts/`, and runs the FastAPI server on port 7860.

---

## 11. References

This model draws from:

- **Extreme Value Theory:** Radial-angular decomposition, directional thresholds
- **Marked Point Processes:** Event-based representation of extremes
- **Hawkes Processes:** Self-exciting temporal dynamics with directional attenuation
- **von Mises-Fisher Distribution:** Directional statistics on the sphere
- **Ogata Thinning:** Exact simulation of point processes
- **Transformers:** History-dependent prediction via causal attention

The combination allows learning and generating realistic cascading extreme events while maintaining interpretability through the cascade probability decomposition and immigration-branching genealogy.

---

(c) 2024-2026 De Carvalho, Ferrer & Vallejos
