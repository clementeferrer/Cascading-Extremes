"""Genealogy module: parent variables, cluster extraction, cascade probabilities.

Implements the immigration-branching structure underlying the Hawkes process.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class Genealogy:
    """Genealogy of a cascade event sequence.

    Attributes
    ----------
    parents : np.ndarray of int, shape (n_events,)
        Parent index for each event. P_i = 0 means immigrant (no parent in history).
        P_i = j > 0 means event j triggered event i (1-indexed in the original history).
        We use -1 for immigrants for clarity.
    immigrant_mask : np.ndarray of bool
        True for immigrant events (exogenous shocks).
    clusters : dict
        Mapping from immigrant index to list of descendant indices.
    cascade_probs : np.ndarray
        Probability each event is endogenous: psi_i / lambda_i.
    """
    parents: np.ndarray
    immigrant_mask: np.ndarray
    clusters: Dict[int, List[int]]
    cascade_probs: np.ndarray


def compute_parent_probabilities(
    lam: np.ndarray,
    mu: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Compute parent probabilities for each event.

    Parameters
    ----------
    lam : (n,) total intensity at each event
    mu : (n,) baseline (exogenous) intensity at each event
    kernel : (n, n) excitation kernel, kernel[i, j] = contribution of j to i

    Returns
    -------
    parent_probs : (n, n+1) matrix where
        parent_probs[i, 0] = P(event i is immigrant) = mu[i] / lam[i]
        parent_probs[i, j+1] = P(parent of i is j) = kernel[i, j] / lam[i]
    """
    n = len(lam)
    probs = np.zeros((n, n + 1), dtype=np.float64)
    lam_safe = np.maximum(lam, 1e-12)

    probs[:, 0] = mu / lam_safe
    for i in range(n):
        probs[i, 1:] = kernel[i, :] / lam_safe[i]

    # Normalize rows to sum to 1
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    probs = probs / row_sums
    return probs


def sample_parents(parent_probs: np.ndarray) -> np.ndarray:
    """Stochastically sample parent assignments from parent probabilities.

    Returns
    -------
    parents : (n,) array of int.
        parents[i] = -1 if immigrant, else index j of parent event.
    """
    n = parent_probs.shape[0]
    parents = np.zeros(n, dtype=np.int64)
    for i in range(n):
        chosen = np.random.choice(parent_probs.shape[1], p=parent_probs[i])
        if chosen == 0:
            parents[i] = -1  # immigrant
        else:
            parents[i] = chosen - 1  # 0-indexed event index
    return parents


def extract_clusters(parents: np.ndarray) -> Dict[int, List[int]]:
    """Extract clusters from parent assignments.

    A cluster is rooted at an immigrant event and contains all its descendants.

    Returns
    -------
    clusters : dict mapping immigrant index -> sorted list of descendant indices (inclusive).
    """
    n = len(parents)
    # Build children map
    children: Dict[int, List[int]] = {}
    immigrants = []
    for i in range(n):
        if parents[i] == -1:
            immigrants.append(i)
        else:
            p = int(parents[i])
            if p not in children:
                children[p] = []
            children[p].append(i)

    clusters = {}
    for imm in immigrants:
        # BFS/DFS to collect all descendants
        cluster = [imm]
        queue = [imm]
        while queue:
            node = queue.pop(0)
            for child in children.get(node, []):
                cluster.append(child)
                queue.append(child)
        clusters[imm] = sorted(cluster)

    return clusters


def build_genealogy(
    lam: np.ndarray,
    psi: np.ndarray,
    kernel: np.ndarray,
) -> Genealogy:
    """Build full genealogy from Hawkes intensity decomposition.

    Parameters
    ----------
    lam : (n,) total intensity
    psi : (n,) endogenous intensity
    kernel : (n, n) excitation kernel matrix

    Returns
    -------
    Genealogy dataclass with parents, clusters, cascade probabilities.
    """
    mu = lam - psi
    mu = np.maximum(mu, 0.0)

    cascade_probs = psi / np.maximum(lam, 1e-12)

    parent_probs = compute_parent_probabilities(lam, mu, kernel)
    parents = sample_parents(parent_probs)
    immigrant_mask = parents == -1
    clusters = extract_clusters(parents)

    return Genealogy(
        parents=parents,
        immigrant_mask=immigrant_mask,
        clusters=clusters,
        cascade_probs=cascade_probs,
    )
