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

The system learns from historical extreme events and generates realistic synthetic cascades that preserve the statistical and geometric properties of real market crashes.

---

## 1. Data Representation

### 1.1 Standardization to Exponential Margins

Raw asset returns are transformed to have **standard exponential margins**:

1. Fit GARCH(1,1) with Student-t innovations to each asset's log-returns
2. Compute standardized residuals: `z_t^j = (r_t^j - μ_t^j) / σ_t^j`
3. Apply the Probability Integral Transform (PIT):

```
Y_t^j = -log(1 - F̂_j(z_t^j))
```

This yields `X_t = (Y_t^1, ..., Y_t^d)` with standard exponential margins.

### 1.2 Radial-Angular Decomposition

Each observation is decomposed into:

- **Radial component** (magnitude): `R_t = ||X_t||`
- **Angular component** (direction): `W_t = X_t / R_t ∈ S^{d-1}_+`

The direction `W` lives on the positive unit simplex, capturing which assets are most extreme relative to each other.

**Implementation:** See `cascades/dataset.py` for the `radial_angular()` function.

---

## 2. Extreme Events as a Marked Point Process

### 2.1 Directional Threshold

An extreme event occurs when:

```
R_t > u_τ(W_t)
```

where `u_τ(w)` is a **direction-dependent quantile threshold** learned via conditional quantile regression. This captures that extremes in different asset combinations may require different magnitude thresholds.

**Implementation:** The `DirectionalQuantileMLP` in `cascades/extremes.py` learns `u_τ(w)`.

### 2.2 Event Sequence

The sequence of extreme events forms a marked point process:

```
N = Σ_i δ_{(T_i, M_i)}
```

where:
- `T_i` = time of event i
- `M_i = (W_{T_i}, R_{T_i})` = mark (direction and magnitude)

---

## 3. The Cascading Extremes Transformer

### 3.1 Token Representation

Each event is encoded as a token:

```
Φ_i = (W_{T_i}, ζ(W_{T_i}), log R_{T_i}, log ΔT_i)
```

where:
- `W` = simplex coordinates (direction)
- `ζ(W)` = geometric features of direction (see `cascades/dataset.py:zeta()`)
- `log R` = log-magnitude
- `log ΔT` = log inter-arrival time

### 3.2 Causal Transformer Encoder

A **causal (autoregressive) transformer** maps the event history to latent states:

```
h_i = Transformer(Φ_{1:i})
```

The transformer uses:
- Positional embeddings
- Causal attention mask (each position only attends to past positions)
- Multiple self-attention layers

**Implementation:** See `CascadingTransformer.encode()` in `cascades/model.py`.

---

## 4. Three-Head Generative Model

Given the latent state `h_i`, three heads predict the next event.

### 4.1 Direction Head (Mixture of Dirichlet)

The next direction is sampled from a **mixture of Dirichlet distributions**:

```
W_{i+1} | H_i ~ Σ_k π_{i,k} · Dir(α_{i,k})
```

where:
- `π_i = softmax(Π(h_i))` = mixture weights
- `α_{i,k} = softplus(A_k(h_i))` = concentration parameters

This allows multi-modal directional predictions (e.g., "next extreme could be BTC-dominant or ETH-dominant").

**Implementation:** See `dirichlet_params()` and `sample_dirichlet_mixture()` in `cascades/model.py` and `cascades/simulate.py`.

### 4.2 Magnitude Head (Truncated Gamma)

Given the predicted direction, the magnitude follows a **truncated Gamma distribution**:

```
R_{i+1} | {W_{i+1}, R > u} ~ TruncGamma(a, β_{i+1})
```

where:
- `a` = learned shape parameter
- `β_{i+1} = softplus(Υ(h_i, log R_i, W_{i+1}))` = rate (gauge function)
- Truncation at `u = u_τ(W_{i+1})` ensures we only generate events exceeding the threshold

**Implementation:** See `gauge_rate()` and `sample_trunc_gamma()` in the model/simulate modules.

### 4.3 Time Head (Hawkes Intensity)

The inter-arrival time follows an **exponential distribution** with Hawkes intensity:

```
λ_i = μ(h_i) + Σ_{j≤i} κ(T_i - T_j, h_i, h_j) · φ(R_j)
```

where:
- `μ(h_i)` = baseline (exogenous) intensity
- `κ(Δ, h_i, h_j)` = excitation kernel with learned magnitude and decay
- `φ(R)` = magnitude impact function

The kernel is:

```
κ(Δ, h_i, h_j) = softplus(k([h_i, h_j])) · exp(-Δ / softplus(τ([h_i, h_j])))
```

**Implementation:** See `hawkes_intensity()` in `cascades/model.py`.

---

## 5. Cascade Probability

A key interpretable quantity is the **cascade probability**:

```
P(cascade | event i) = ψ_i / λ_i
```

where:
- `ψ_i = Σ_j κ(·) · φ(R_j)` = endogenous (self-excited) component
- `λ_i = μ_i + ψ_i` = total intensity

When `ψ/λ` is high, the event was likely triggered by previous events (a cascade). When low, it's an exogenous shock.

**Visualization:** The immersive viewer colors points by cascade probability (viridis scale: blue=exogenous, yellow=cascade).

---

## 6. Training Objective

The model is trained by maximizing the joint log-likelihood:

```
L = Σ_i [ log p(W_{i+1}|h_i) + log p(R_{i+1}|h_i) + log p(ΔT_{i+1}|h_i) ]
```

Each component:
- **Direction:** Log-likelihood under the Dirichlet mixture
- **Magnitude:** Truncated Gamma log-density
- **Time:** Exponential log-density: `log λ - λ · ΔT`

**Implementation:** See `log_likelihood()` in `cascades/model.py` and `cascades/train.py`.

---

## 7. Generation Algorithm

The `generate_with_limits()` function in `cascades/simulate.py` implements autoregressive generation:

```python
for each new event:
    1. Encode history: h = Transformer(tokens)

    2. Sample direction:
       - Get mixture params: π, α = dirichlet_params(h)
       - Sample component k ~ Categorical(π)
       - Sample W ~ Dirichlet(α_k)

    3. Sample magnitude:
       - Get rate: β = gauge_rate(h, log R_prev, W)
       - Get threshold: u = quantile_model(W)
       - Sample R ~ TruncGamma(a, β; R > u)

    4. Sample timing:
       - Get intensity: λ = hawkes_intensity(h, T, log R)
       - Sample ΔT ~ Exponential(λ)
       - T_new = T_prev + ΔT

    5. Stop if T_new > max_time or num_events > max_events
```

**Key constraints enforced:**
- `max_events`: Maximum number of events to generate
- `max_time`: Horizon beyond which generation stops (relative to seed)

---

## 8. Project Structure

```
cascades/
├── dataset.py      # Data loading, tokenization, radial-angular transform
├── extremes.py     # Directional quantile model for thresholds
├── model.py        # CascadingTransformer with three heads
├── train.py        # Training loop and loss computation
├── simulate.py     # Autoregressive generation algorithm
└── utils.py        # Configuration, seeding, I/O utilities

web/
├── api/main.py     # FastAPI backend for viewer
└── immersive/      # React + Three.js visualization
    └── src/
        ├── pages/ViewerPage.tsx    # Main viewer component
        ├── scenes/CascadeScene.tsx # 3D scene with simplex
        └── scenes/EventsPoints.tsx # Point cloud colored by ψ/λ

app.py              # Streamlit dashboard for analysis
```

---

## 9. Key Equations Summary

| Component | Distribution | Parameters from Transformer |
|-----------|-------------|----------------------------|
| Direction W | Mixture of Dirichlet | π (mixture weights), α (concentrations) |
| Magnitude R | Truncated Gamma | a (shape), β (rate from gauge network) |
| Time ΔT | Exponential | λ (Hawkes intensity) |
| Cascade Prob | ψ/λ | Ratio of endogenous to total intensity |

---

## 10. References

This model draws from:

- **Extreme Value Theory:** Radial-angular decomposition, directional thresholds
- **Marked Point Processes:** Event-based representation of extremes
- **Hawkes Processes:** Self-exciting temporal dynamics
- **Transformers:** History-dependent prediction via causal attention

The combination allows learning and generating realistic cascading extreme events while maintaining interpretability through the cascade probability decomposition.

---

© 2024-2026 De Carvalho, Ferrer & Vallejos
