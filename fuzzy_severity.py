# filename: fuzzy_severity.py
"""
Mamdani Fuzzy Inference System (FIS) for Disaster Zone Severity Assessment.
Uses scikit-fuzzy with 4 antecedents, 1 consequent, and 22 domain rules.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def _build_fis():
    """Construct the Mamdani ControlSystem and rule base."""
    u_in = np.linspace(0, 10, 101)
    u_out = np.linspace(0, 100, 201)

    damage_level = ctrl.Antecedent(u_in, "damage_level")
    population_density = ctrl.Antecedent(u_in, "population_density")
    accessibility = ctrl.Antecedent(u_in, "accessibility")
    casualty_count = ctrl.Antecedent(u_in, "casualty_count")
    severity = ctrl.Consequent(u_out, "severity", defuzzify_method="centroid")

    # Antecedent membership functions: Low (trapmf), Medium (trimf), High (trapmf)
    for ant in (damage_level, population_density, accessibility, casualty_count):
        ant["Low"] = fuzz.trapmf(u_in, [0, 0, 3, 5])
        ant["Medium"] = fuzz.trimf(u_in, [3, 5, 7])
        ant["High"] = fuzz.trapmf(u_in, [5, 7, 10, 10])

    # Consequent membership functions
    severity["Low"] = fuzz.trapmf(u_out, [0, 0, 25, 40])
    severity["Moderate"] = fuzz.trapmf(u_out, [25, 40, 55, 70])
    severity["High"] = fuzz.trapmf(u_out, [55, 70, 85, 95])
    severity["Critical"] = fuzz.trapmf(u_out, [80, 95, 100, 100])

    # 22 Domain Rules (covering Low, Moderate, High, Critical at least 4 times each)
    rules_spec = [
        # Critical rules (6)
        ("R01: Dmg=High & Cas=High -> Critical",
         damage_level["High"] & casualty_count["High"], "Critical",
         lambda d, p, a, c: min(d["High"], c["High"])),
        ("R02: Dmg=High & Pop=High & Acc=Low -> Critical",
         damage_level["High"] & population_density["High"] & accessibility["Low"], "Critical",
         lambda d, p, a, c: min(d["High"], p["High"], a["Low"])),
        ("R03: Cas=High & Acc=Low -> Critical",
         casualty_count["High"] & accessibility["Low"], "Critical",
         lambda d, p, a, c: min(c["High"], a["Low"])),
        ("R04: Dmg=High & Cas=Med & Acc=Low -> Critical",
         damage_level["High"] & casualty_count["Medium"] & accessibility["Low"], "Critical",
         lambda d, p, a, c: min(d["High"], c["Medium"], a["Low"])),
        ("R05: Dmg=Med & Cas=High & Pop=High -> Critical",
         damage_level["Medium"] & casualty_count["High"] & population_density["High"], "Critical",
         lambda d, p, a, c: min(d["Medium"], c["High"], p["High"])),
        ("R06: Cas=High & Pop=High & Acc=Med -> Critical",
         casualty_count["High"] & population_density["High"] & accessibility["Medium"], "Critical",
         lambda d, p, a, c: min(c["High"], p["High"], a["Medium"])),

        # High rules (6)
        ("R07: Acc=Low & Dmg=High -> High",
         accessibility["Low"] & damage_level["High"], "High",
         lambda d, p, a, c: min(a["Low"], d["High"])),
        ("R08: Dmg=High & Cas=Med -> High",
         damage_level["High"] & casualty_count["Medium"], "High",
         lambda d, p, a, c: min(d["High"], c["Medium"])),
        ("R09: Dmg=Med & Cas=High -> High",
         damage_level["Medium"] & casualty_count["High"], "High",
         lambda d, p, a, c: min(d["Medium"], c["High"])),
        ("R10: Dmg=High & Pop=High & Acc=Med -> High",
         damage_level["High"] & population_density["High"] & accessibility["Medium"], "High",
         lambda d, p, a, c: min(d["High"], p["High"], a["Medium"])),
        ("R11: Dmg=Med & Pop=High & Acc=Low -> High",
         damage_level["Medium"] & population_density["High"] & accessibility["Low"], "High",
         lambda d, p, a, c: min(d["Medium"], p["High"], a["Low"])),
        ("R12: Cas=High & Acc=High & Dmg=Low -> High",
         casualty_count["High"] & accessibility["High"] & damage_level["Low"], "High",
         lambda d, p, a, c: min(c["High"], a["High"], d["Low"])),

        # Moderate rules (5)
        ("R13: Dmg=Med & Cas=Med -> Moderate",
         damage_level["Medium"] & casualty_count["Medium"], "Moderate",
         lambda d, p, a, c: min(d["Medium"], c["Medium"])),
        ("R14: Dmg=High & Cas=Low & Acc=High -> Moderate",
         damage_level["High"] & casualty_count["Low"] & accessibility["High"], "Moderate",
         lambda d, p, a, c: min(d["High"], c["Low"], a["High"])),
        ("R15: Dmg=Low & Cas=Med & Pop=High -> Moderate",
         damage_level["Low"] & casualty_count["Medium"] & population_density["High"], "Moderate",
         lambda d, p, a, c: min(d["Low"], c["Medium"], p["High"])),
        ("R16: Dmg=Med & Cas=Low & Acc=Low -> Moderate",
         damage_level["Medium"] & casualty_count["Low"] & accessibility["Low"], "Moderate",
         lambda d, p, a, c: min(d["Medium"], c["Low"], a["Low"])),
        ("R17: Pop=Med & Acc=Med & Dmg=Med -> Moderate",
         population_density["Medium"] & accessibility["Medium"] & damage_level["Medium"], "Moderate",
         lambda d, p, a, c: min(p["Medium"], a["Medium"], d["Medium"])),

        # Low rules (5)
        ("R18: Dmg=Low & Cas=Low -> Low",
         damage_level["Low"] & casualty_count["Low"], "Low",
         lambda d, p, a, c: min(d["Low"], c["Low"])),
        ("R19: Dmg=Low & Acc=High & Cas=Low -> Low",
         damage_level["Low"] & accessibility["High"] & casualty_count["Low"], "Low",
         lambda d, p, a, c: min(d["Low"], a["High"], c["Low"])),
        ("R20: Dmg=Low & Pop=Low & Cas=Med -> Low",
         damage_level["Low"] & population_density["Low"] & casualty_count["Medium"], "Low",
         lambda d, p, a, c: min(d["Low"], p["Low"], c["Medium"])),
        ("R21: Dmg=Med & Cas=Low & Acc=High & Pop=Low -> Low",
         damage_level["Medium"] & casualty_count["Low"] & accessibility["High"] & population_density["Low"], "Low",
         lambda d, p, a, c: min(d["Medium"], c["Low"], a["High"], p["Low"])),
        ("R22: Pop=Low & Acc=High & Cas=Low -> Low",
         population_density["Low"] & accessibility["High"] & casualty_count["Low"], "Low",
         lambda d, p, a, c: min(p["Low"], a["High"], c["Low"])),
    ]

    ctrl_rules = [
        ctrl.Rule(antecedent=expr, consequent=severity[out_term], label=label)
        for label, expr, out_term, _ in rules_spec
    ]
    system = ctrl.ControlSystem(ctrl_rules)
    antecedents = {
        "damage_level": damage_level,
        "population_density": population_density,
        "accessibility": accessibility,
        "casualty_count": casualty_count,
    }
    return system, antecedents, severity, rules_spec


_SYSTEM, _ANTECEDENTS, _SEVERITY, _RULES_SPEC = _build_fis()


def categorise(score: float) -> str:
    """Categorise continuous severity score (0-100) using thresholds 40/60/80."""
    if score < 40.0:
        return "Low"
    if score < 60.0:
        return "Moderate"
    if score < 80.0:
        return "High"
    return "Critical"


def _compute_memberships(val: float) -> dict:
    """Compute Low, Medium, High membership degrees for a scalar in [0, 10]."""
    u = np.linspace(0, 10, 101)
    return {
        "Low": float(fuzz.interp_membership(u, fuzz.trapmf(u, [0, 0, 3, 5]), val)),
        "Medium": float(fuzz.interp_membership(u, fuzz.trimf(u, [3, 5, 7]), val)),
        "High": float(fuzz.interp_membership(u, fuzz.trapmf(u, [5, 7, 10, 10]), val)),
    }


def assess_severity(
    damage: float, pop_density: float, accessibility: float, casualties: float
) -> dict:
    """
    Evaluate Mamdani FIS for a zone's 4 inputs and return score, category, and rule strengths.
    """
    sim = ctrl.ControlSystemSimulation(_SYSTEM)
    sim.input["damage_level"] = float(np.clip(damage, 0.0, 10.0))
    sim.input["population_density"] = float(np.clip(pop_density, 0.0, 10.0))
    sim.input["accessibility"] = float(np.clip(accessibility, 0.0, 10.0))
    sim.input["casualty_count"] = float(np.clip(casualties, 0.0, 10.0))

    try:
        sim.compute()
        score = float(sim.output["severity"])
    except Exception:
        score = 50.0

    # Calculate exact firing strength per rule for explainability
    d_m = _compute_memberships(damage)
    p_m = _compute_memberships(pop_density)
    a_m = _compute_memberships(accessibility)
    c_m = _compute_memberships(casualties)

    rule_strengths = {
        label: round(float(fn(d_m, p_m, a_m, c_m)), 4)
        for label, _, _, fn in _RULES_SPEC
    }

    return {
        "score": round(score, 2),
        "category": categorise(score),
        "rule_strengths": rule_strengths,
    }


def plot_membership_functions() -> plt.Figure:
    """Plot the 4 antecedent membership functions and consequent severity in a 4-subplot figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    u_in = _ANTECEDENTS["damage_level"].universe
    u_out = _SEVERITY.universe

    configs = [
        (axes[0, 0], _ANTECEDENTS["damage_level"], "Damage Level & Pop. Density (0-10)", u_in),
        (axes[0, 1], _ANTECEDENTS["accessibility"], "Accessibility Index (0-10)", u_in),
        (axes[1, 0], _ANTECEDENTS["casualty_count"], "Casualty Count Index (0-10)", u_in),
        (axes[1, 1], _SEVERITY, "Consequent: Disaster Severity (0-100)", u_out),
    ]

    colors = {"Low": "#2ecc71", "Medium": "#f1c40f", "Moderate": "#f39c12", "High": "#e67e22", "Critical": "#e74c3c"}

    for ax, var, title, univ in configs:
        for term_name, term_obj in var.terms.items():
            ax.plot(univ, term_obj.mf, label=term_name, linewidth=2.2, color=colors.get(term_name))
            ax.fill_between(univ, 0, term_obj.mf, alpha=0.15, color=colors.get(term_name))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Universe of Discourse")
        ax.set_ylabel("Membership Degree μ(x)")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Mamdani Fuzzy Membership Functions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_severity_surface(grid_res: int = 21) -> go.Figure:
    """Generate 3D Plotly response surface for Severity vs Damage × Casualties (others=5)."""
    vals = np.linspace(0, 10, grid_res)
    z_matrix = np.zeros((grid_res, grid_res))

    for i, cas in enumerate(vals):
        for j, dmg in enumerate(vals):
            z_matrix[i, j] = assess_severity(dmg, 5.0, 5.0, cas)["score"]

    fig = go.Figure(
        data=[
            go.Surface(
                x=vals,
                y=vals,
                z=z_matrix,
                colorscale="YlOrRd",
                colorbar=dict(title="Severity"),
            )
        ]
    )
    fig.update_layout(
        title="3D Mamdani Control Surface: Damage × Casualties (Density=5, Access=5)",
        scene=dict(
            xaxis_title="Damage Level (0-10)",
            yaxis_title="Casualty Count (0-10)",
            zaxis_title="Fuzzy Severity (0-100)",
        ),
        margin=dict(l=10, r=10, b=10, t=45),
        height=500,
    )
    return fig