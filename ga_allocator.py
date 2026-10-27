# filename: ga_allocator.py
"""
Custom Genetic Algorithm for Multi-Resource Disaster Allocation.
Includes integer chromosome encoding, constraint repair, tournament selection,
uniform crossover, random-reset mutation, and elitism.
"""

import numpy as np
import pandas as pd
from baselines import RESOURCE_COLS, coverage_score


class GAAllocator:
    """
    Hand-rolled Genetic Algorithm optimizing integer resource allocation across zones.
    Chromosome layout: flat array of length (n_zones * 3) reshaped as (n_zones, 3).
    """

    def __init__(
        self,
        severities: np.ndarray | list,
        resources: dict,
        demands: pd.DataFrame,
        population: np.ndarray | list,
        pop_size: int = 60,
        generations: int = 80,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.85,
        seed: int = 42,
    ):
        self.severities = np.asarray(severities, dtype=float)
        self.resources = resources
        self.demands = demands.copy()
        self.population_vec = np.asarray(population, dtype=float)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.n_zones = len(self.severities)
        self.n_res = len(RESOURCE_COLS)
        self.supply_vec = np.array([int(resources[c]) for c in RESOURCE_COLS], dtype=int)
        self.demand_matrix = demands[RESOURCE_COLS].to_numpy(dtype=int)
        self.total_demand_norm = float(np.sum(self.demand_matrix)) + 1e-9

    def _chromosome_to_df(self, chrom: np.ndarray) -> pd.DataFrame:
        """Convert flat chromosome of length (n_zones * 3) into an allocation DataFrame."""
        mat = chrom.reshape((self.n_zones, self.n_res))
        return pd.DataFrame({
            "zone_id": self.demands["zone_id"].to_numpy(),
            "ambulances": mat[:, 0],
            "food_kits": mat[:, 1],
            "medical_teams": mat[:, 2],
        })

    def _repair(self, chrom: np.ndarray) -> np.ndarray:
        """
        Enforce feasibility constraints on chromosome:
        1. 0 <= gene <= zone_demand
        2. Sum of allocated units per resource <= available supply (and utilize idle supply).
        """
        mat = np.clip(chrom.reshape((self.n_zones, self.n_res)).copy(), 0, self.demand_matrix)

        for r_idx in range(self.n_res):
            supply = self.supply_vec[r_idx]
            col = mat[:, r_idx]
            current_sum = int(np.sum(col))

            # Reduce excess if over-allocated
            while current_sum > supply:
                nonzero = np.where(col > 0)[0]
                if nonzero.size == 0:
                    break
                # Prefer removing from lower-severity zones first
                inv_weights = 1.0 / (self.severities[nonzero] + 5.0)
                probs = inv_weights / np.sum(inv_weights)
                pick = int(self.rng.choice(nonzero, p=probs))
                step = min(int(col[pick]), max(1, (current_sum - supply) // 3))
                col[pick] -= step
                current_sum -= step

            # Distribute unused supply to zones with remaining demand
            while current_sum < supply:
                under = np.where(col < self.demand_matrix[:, r_idx])[0]
                if under.size == 0:
                    break
                weights = self.severities[under] + 1.0
                probs = weights / np.sum(weights)
                pick = int(self.rng.choice(under, p=probs))
                room = int(self.demand_matrix[pick, r_idx] - col[pick])
                step = min(room, max(1, (supply - current_sum) // 3))
                col[pick] += step
                current_sum += step

            mat[:, r_idx] = col

        return mat.flatten()

    def _init_population(self) -> list[np.ndarray]:
        """Initialize diverse population of repaired integer chromosomes."""
        pop = []
        for _ in range(self.pop_size):
            rand_mat = self.rng.uniform(0.0, 1.0, size=(self.n_zones, self.n_res))
            raw_chrom = np.round(rand_mat * self.demand_matrix).astype(int).flatten()
            pop.append(self._repair(raw_chrom))
        return pop

    def _fitness(self, chrom: np.ndarray) -> float:
        """
        Compute fitness:
        coverage_score['total_score'] - 0.5 * unmet_demand_penalty - 2.0 * resource_overuse_penalty
        Also includes a soft harmonic bonus for balanced coverage across zones.
        """
        df_alloc = self._chromosome_to_df(chrom)
        metrics = coverage_score(df_alloc, self.severities, self.demands, self.population_vec)

        mat = chrom.reshape((self.n_zones, self.n_res))
        overuse = float(np.sum(np.maximum(0, np.sum(mat, axis=0) - self.supply_vec)))

        # Normalized unmet demand penalty so scale aligns with total_score
        unmet_norm = (metrics["unmet_demand"] / self.total_demand_norm) * 20.0
        # Diminishing-returns balance bonus (rewards avoiding 0-allocation to critical zones)
        ratios = np.minimum(1.0, mat / np.maximum(1.0, self.demand_matrix))
        balance_bonus = float(np.sum(self.severities * np.sqrt(np.mean(ratios, axis=1))) * 0.12)

        return float(metrics["total_score"] + balance_bonus - 0.5 * unmet_norm - 2.0 * overuse)

    def _tournament(self, population: list[np.ndarray], fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
        """Select parent using k-way tournament selection."""
        idxs = self.rng.integers(0, len(population), size=k)
        best_i = idxs[int(np.argmax(fitnesses[idxs]))]
        return population[best_i].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uniform crossover exchanging genes independently with probability 0.5."""
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        mask = self.rng.random(size=p1.size) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _mutate(self, chrom: np.ndarray) -> np.ndarray:
        """Random-reset and swap mutation per gene."""
        mat = chrom.reshape((self.n_zones, self.n_res)).copy()
        for z in range(self.n_zones):
            for r in range(self.n_res):
                if self.rng.random() < self.mutation_rate:
                    max_dem = int(self.demand_matrix[z, r])
                    mat[z, r] = int(self.rng.integers(0, max_dem + 1))
        return mat.flatten()

    def run(self) -> dict:
        """Execute evolutionary loop and return best solution and convergence history."""
        population = self._init_population()
        fitnesses = np.array([self._fitness(ind) for ind in population])

        best_idx = int(np.argmax(fitnesses))
        best_chrom = population[best_idx].copy()
        best_fit = float(fitnesses[best_idx])
        history = [round(best_fit, 3)]

        for _ in range(self.generations):
            # Elitism: preserve top 2 individuals
            elite_idxs = np.argsort(-fitnesses)[:2]
            new_pop = [population[i].copy() for i in elite_idxs]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._repair(self._mutate(c1))
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    c2 = self._repair(self._mutate(c2))
                    new_pop.append(c2)

            population = new_pop
            fitnesses = np.array([self._fitness(ind) for ind in population])

            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_fit:
                best_fit = float(fitnesses[gen_best_idx])
                best_chrom = population[gen_best_idx].copy()

            history.append(round(best_fit, 3))

        return {
            "best_chromosome": best_chrom,
            "best_fitness": round(best_fit, 2),
            "history": history,
            "allocation_df": self._chromosome_to_df(best_chrom),
        }