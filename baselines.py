# filename: baselines.py
"""
Baseline Comparison Models & Multi-Metric Evaluation for Disaster Resource Allocation.
Implements:
  1. Fixed-weight linear severity baseline (threshold_severity)
  2. Greedy resource allocator (prioritizing top-severity zones)
  3. Proportional resource allocator (distributing proportionally to severity & demand)
  4. Random resource allocator (reproducible randomized allocation)
  5. Multi-metric evaluation (coverage_score: total score, unmet demand, equity Gini, per-zone coverage)
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

RESOURCE_COLS = ["ambulances", "food_kits", "medical_teams"]


def threshold_severity(
    damage_level: float,
    population_density: float,
    accessibility: float,
    casualty_count: float,
) -> float:
    """
    Fixed-weight linear severity assessment baseline.
    Weights: [0.35, 0.20, 0.15, 0.30] corresponding to
    damage_level, population_density, inaccessibility (10 - accessibility), and casualty_count.
    All inputs are on a [0, 10] scale.

    Returns
    -------
    float
        Severity score on a [0, 100] scale.
    """
    inaccessibility = 10.0 - float(accessibility)
    score = (
        0.35 * float(damage_level)
        + 0.20 * float(population_density)
        + 0.15 * inaccessibility
        + 0.30 * float(casualty_count)
    ) * 10.0
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def coverage_score(
    alloc_df: pd.DataFrame,
    severities: Union[np.ndarray, pd.Series, List[float]],
    demands: pd.DataFrame,
    population: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Evaluate resource allocation across zones using multi-dimensional criteria.

    Parameters
    ----------
    alloc_df : pd.DataFrame
        Resource allocation DataFrame containing RESOURCE_COLS.
    severities : array-like
        Severity score per zone (0 to 100).
    demands : pd.DataFrame
        Demanded units per zone containing RESOURCE_COLS and 'zone_id'.
    population : array-like, optional
        Population per zone for population-weighted scoring.

    Returns
    -------
    dict with keys:
        - 'total_score': Weighted coverage score (float)
        - 'unmet_demand': Total unmet resource units across all zones (int)
        - 'equity_gini': Gini coefficient measuring inequality in zone coverage (float)
        - 'per_zone_coverage': Dict mapping zone_id -> coverage percentage [0, 100] (float)
    """
    alloc_mat = alloc_df[RESOURCE_COLS].to_numpy(dtype=float)
    demand_mat = demands[RESOURCE_COLS].to_numpy(dtype=float)
    sev_arr = np.asarray(severities, dtype=float)
    n_zones = len(sev_arr)

    # 1. Total unmet demand in absolute resource units
    unmet_demand = int(np.sum(np.maximum(0.0, demand_mat - alloc_mat)))

    # 2. Per-resource and per-zone coverage ratios [0.0, 1.0]
    safe_demand = np.maximum(1.0, demand_mat)
    cov_ratio_mat = np.clip(alloc_mat / safe_demand, 0.0, 1.0)
    zone_cov_ratios = np.mean(cov_ratio_mat, axis=1)

    # Zone ID mapping for display
    zone_ids = demands["zone_id"].tolist() if "zone_id" in demands.columns else [f"Z{i+1:02d}" for i in range(n_zones)]
    per_zone_coverage = {
        z_id: round(float(cov * 100.0), 2)
        for z_id, cov in zip(zone_ids, zone_cov_ratios)
    }

    # 3. Total Weighted Coverage Score
    if population is not None:
        pop_arr = np.asarray(population, dtype=float)
        pop_max = np.max(pop_arr) if np.max(pop_arr) > 0 else 1.0
        pop_norm = pop_arr / pop_max
        urgency_weights = (sev_arr / 100.0) * (0.45 + 0.55 * pop_norm)
    else:
        urgency_weights = sev_arr / 100.0

    # Scale total_score by 100 so it aligns with fitness and display scales
    total_score = round(float(np.sum(urgency_weights * zone_cov_ratios) * 100.0), 2)

    # 4. Equity Gini Coefficient of coverage across zones
    # 0 = perfect equality, 1 = maximum inequality
    y = np.sort(zone_cov_ratios)
    y_sum = np.sum(y)
    if y_sum <= 1e-9:
        equity_gini = 0.0
    else:
        ranks = np.arange(1, n_zones + 1)
        gini = (2.0 * np.sum(ranks * y)) / (n_zones * y_sum) - (n_zones + 1.0) / n_zones
        equity_gini = round(float(np.clip(gini, 0.0, 1.0)), 4)

    return {
        "total_score": total_score,
        "unmet_demand": unmet_demand,
        "equity_gini": equity_gini,
        "per_zone_coverage": per_zone_coverage,
    }


def greedy_allocation(
    severities: Union[np.ndarray, pd.Series, List[float]],
    resources: dict,
    demands: pd.DataFrame,
) -> pd.DataFrame:
    """
    Greedy baseline allocator: prioritizes zones in strict descending order of severity.
    Completely satisfies demands of highest-severity zones first until supplies are depleted.
    """
    sev_arr = np.asarray(severities, dtype=float)
    n_zones = len(sev_arr)
    alloc_mat = np.zeros((n_zones, len(RESOURCE_COLS)), dtype=int)
    sort_order = np.argsort(-sev_arr)

    for r_idx, col in enumerate(RESOURCE_COLS):
        supply = int(resources[col])
        demand_col = demands[col].to_numpy(dtype=int)
        for z_idx in sort_order:
            if supply <= 0:
                break
            needed = demand_col[z_idx]
            granted = min(supply, needed)
            alloc_mat[z_idx, r_idx] = granted
            supply -= granted

    zone_ids = demands["zone_id"].to_numpy() if "zone_id" in demands.columns else np.array([f"Z{i+1:02d}" for i in range(n_zones)])
    data = {"zone_id": zone_ids}
    for i, col in enumerate(RESOURCE_COLS):
        data[col] = alloc_mat[:, i]

    return pd.DataFrame(data)


def proportional_allocation(
    severities: Union[np.ndarray, pd.Series, List[float]],
    resources: dict,
    demands: pd.DataFrame,
) -> pd.DataFrame:
    """
    Proportional baseline allocator: distributes scarce resources across all zones
    in proportion to urgency (severity * demand), using Hamilton's largest-remainder method.
    """
    sev_arr = np.asarray(severities, dtype=float)
    n_zones = len(sev_arr)
    alloc_mat = np.zeros((n_zones, len(RESOURCE_COLS)), dtype=int)

    for r_idx, col in enumerate(RESOURCE_COLS):
        supply = int(resources[col])
        demand_col = demands[col].to_numpy(dtype=int)
        tot_demand = int(np.sum(demand_col))

        if tot_demand <= supply:
            alloc_mat[:, r_idx] = demand_col
            continue

        # Urgency weight per zone for this resource
        weights = (sev_arr + 1.0) * demand_col
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            quotas = supply * (weights / w_sum)
        else:
            quotas = supply * (demand_col / float(tot_demand))

        # Cap quotas at demand
        quotas = np.minimum(quotas, demand_col.astype(float))
        base = np.floor(quotas).astype(int)
        rem = supply - int(np.sum(base))

        # Largest-remainder distribution
        remainders = quotas - base
        order = np.argsort(-remainders)
        for idx in order:
            if rem <= 0:
                break
            if base[idx] < demand_col[idx]:
                base[idx] += 1
                rem -= 1

        # Fallback distribution for any remaining units
        if rem > 0:
            for idx in range(n_zones):
                if rem <= 0:
                    break
                room = int(demand_col[idx] - base[idx])
                step = min(room, rem)
                base[idx] += step
                rem -= step

        alloc_mat[:, r_idx] = base

    zone_ids = demands["zone_id"].to_numpy() if "zone_id" in demands.columns else np.array([f"Z{i+1:02d}" for i in range(n_zones)])
    data = {"zone_id": zone_ids}
    for i, col in enumerate(RESOURCE_COLS):
        data[col] = alloc_mat[:, i]

    return pd.DataFrame(data)


def random_allocation(
    severities: Union[np.ndarray, pd.Series, List[float]],
    resources: dict,
    demands: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Random baseline allocator: distributes scarce resources randomly across zones
    subject to zone demand and total supply constraints.
    """
    rng = np.random.default_rng(seed)
    sev_arr = np.asarray(severities, dtype=float)
    n_zones = len(sev_arr)
    alloc_mat = np.zeros((n_zones, len(RESOURCE_COLS)), dtype=int)

    for r_idx, col in enumerate(RESOURCE_COLS):
        supply = int(resources[col])
        demand_col = demands[col].to_numpy(dtype=int)
        tot_demand = int(np.sum(demand_col))

        if tot_demand <= supply:
            alloc_mat[:, r_idx] = demand_col
            continue

        # Random weights
        rand_weights = rng.uniform(0.1, 1.0, size=n_zones) * demand_col
        w_sum = float(np.sum(rand_weights))
        if w_sum > 0:
            quotas = supply * (rand_weights / w_sum)
        else:
            quotas = supply / float(n_zones)

        quotas = np.minimum(quotas, demand_col.astype(float))
        base = np.floor(quotas).astype(int)
        rem = supply - int(np.sum(base))

        remainders = quotas - base
        order = np.argsort(-remainders)
        for idx in order:
            if rem <= 0:
                break
            if base[idx] < demand_col[idx]:
                base[idx] += 1
                rem -= 1

        if rem > 0:
            eligible = [i for i in range(n_zones) if base[i] < demand_col[i]]
            while rem > 0 and len(eligible) > 0:
                pick = rng.choice(eligible)
                base[pick] += 1
                rem -= 1
                if base[pick] >= demand_col[pick]:
                    eligible.remove(pick)

        alloc_mat[:, r_idx] = base

    zone_ids = demands["zone_id"].to_numpy() if "zone_id" in demands.columns else np.array([f"Z{i+1:02d}" for i in range(n_zones)])
    data = {"zone_id": zone_ids}
    for i, col in enumerate(RESOURCE_COLS):
        data[col] = alloc_mat[:, i]

    return pd.DataFrame(data)
