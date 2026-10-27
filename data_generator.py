# filename: data_generator.py
"""
Synthetic Disaster Scenario & Resource Demand Generator.
Generates correlated disaster metrics and resource scarcity constraints.
"""

import numpy as np
import pandas as pd


def generate_disaster_scenario(seed: int = 42, n_zones: int = 10) -> pd.DataFrame:
    """
    Generate synthetic disaster zones with realistic correlated attributes.
    - casualty_count correlates positively with damage_level and population_density.
    - accessibility correlates inversely with damage_level.
    """
    rng = np.random.default_rng(seed)

    names = [
        "Alpha-Downtown", "Bravo-Harbor", "Charlie-Heights", "Delta-Valley",
        "Echo-Industrial", "Foxtrot-Suburbs", "Golf-Riverside", "Hotel-OldTown",
        "India-EastEnd", "Juliet-NorthGate", "Kilo-Westside", "Lima-SouthPark",
        "Mike-University", "November-Hilltop", "Oscar-Bayview", "Papa-Midtown",
        "Quebec-Docklands", "Romeo-Greenbelt", "Sierra-Crossroads", "Tango-Outskirts"
    ]

    # Base independent variables
    damage = np.clip(rng.normal(loc=5.8, scale=2.4, size=n_zones), 0.5, 9.8)
    pop_density = np.clip(rng.uniform(1.5, 9.5, size=n_zones), 0.5, 9.8)

    # Correlated variables
    accessibility = np.clip(
        10.0 - 0.72 * damage + rng.normal(0.0, 1.3, size=n_zones), 0.5, 9.8
    )
    casualties = np.clip(
        0.60 * damage + 0.30 * pop_density + rng.normal(0.0, 1.1, size=n_zones),
        0.2, 9.8
    )

    # Scale population realistically with population_density
    raw_pop = 500 + (pop_density / 10.0) * 44000 + rng.integers(-400, 4500, size=n_zones)
    population = np.clip(raw_pop, 500, 50000).astype(int)

    df = pd.DataFrame({
        "zone_id": [f"Z{i+1:02d}" for i in range(n_zones)],
        "zone_name": [names[i % len(names)] for i in range(n_zones)],
        "damage_level": np.round(damage, 2),
        "population_density": np.round(pop_density, 2),
        "accessibility": np.round(accessibility, 2),
        "casualty_count": np.round(casualties, 2),
        "population": population,
    })
    return df


def generate_resources() -> dict:
    """Return available emergency supply pool (scarce by design)."""
    return {
        "ambulances": 8,
        "food_kits": 500,
        "medical_teams": 5,
    }


def generate_zone_demands(df: pd.DataFrame, resources: dict) -> pd.DataFrame:
    """
    Calculate resource demands per zone proportional to population and severity proxy.
    Ensures total demand across all zones exceeds available supply (~1.6x to 2.2x).
    """
    if "fuzzy_severity" in df.columns:
        sev_norm = np.clip(df["fuzzy_severity"].to_numpy() / 100.0, 0.15, 1.0)
    else:
        # Severity proxy from raw metrics (accessibility inverted)
        sev_norm = np.clip(
            (0.35 * df["damage_level"]
             + 0.20 * df["population_density"]
             + 0.15 * (10.0 - df["accessibility"])
             + 0.30 * df["casualty_count"]) / 10.0,
            0.15, 1.0
        )

    pop_norm = df["population"].to_numpy() / df["population"].max()
    urgency_weights = sev_norm * (0.45 + 0.55 * pop_norm)
    weight_share = urgency_weights / urgency_weights.sum()

    # Target ~1.85x total scarcity ratio so allocation is non-trivial
    scarcity_factor = 1.85
    amb_demand = np.maximum(1, np.round(weight_share * resources["ambulances"] * scarcity_factor)).astype(int)
    food_demand = np.maximum(15, np.round(weight_share * resources["food_kits"] * scarcity_factor)).astype(int)
    med_demand = np.maximum(1, np.round(weight_share * resources["medical_teams"] * scarcity_factor)).astype(int)

    return pd.DataFrame({
        "zone_id": df["zone_id"],
        "ambulances": amb_demand,
        "food_kits": food_demand,
        "medical_teams": med_demand,
    })


if __name__ == "__main__":
    scenario_df = generate_disaster_scenario(seed=42, n_zones=10)
    res_dict = generate_resources()
    demands_df = generate_zone_demands(scenario_df, res_dict)

    scenario_df.to_csv("scenario.csv", index=False)
    pd.DataFrame([res_dict]).to_csv("resources.csv", index=False)
    demands_df.to_csv("demands.csv", index=False)
    print("Saved scenario.csv, resources.csv, and demands.csv successfully.")