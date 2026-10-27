# filename: app.py
"""
Streamlit Dashboard for Hybrid Fuzzy-Genetic Disaster Response Framework.
Run with: streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import (
    generate_disaster_scenario,
    generate_resources,
    generate_zone_demands,
)
from fuzzy_severity import (
    assess_severity,
    plot_membership_functions,
    plot_severity_surface,
)
from baselines import (
    threshold_severity,
    greedy_allocation,
    random_allocation,
    proportional_allocation,
    coverage_score,
)
from ga_allocator import GAAllocator

st.set_page_config(
    page_title="Hybrid Fuzzy–GA Disaster Response",
    page_icon="🚨",
    layout="wide",
)

# ─── CACHED PIPELINE FUNCTIONS ───────────────────────────────────────────────
@st.cache_data
def load_scenario_bundle(seed: int, n_zones: int):
    """Generate and cache scenario, fuzzy assessments, threshold baselines, and demands."""
    df = generate_disaster_scenario(seed=seed, n_zones=n_zones)
    resources = generate_resources()

    fuzzy_scores, fuzzy_cats, thresh_scores = [], [], []
    for _, row in df.iterrows():
        res = assess_severity(
            row["damage_level"],
            row["population_density"],
            row["accessibility"],
            row["casualty_count"],
        )
        fuzzy_scores.append(res["score"])
        fuzzy_cats.append(res["category"])
        thresh_scores.append(
            threshold_severity(
                row["damage_level"],
                row["population_density"],
                row["accessibility"],
                row["casualty_count"],
            )
        )

    df["fuzzy_severity"] = fuzzy_scores
    df["severity_category"] = fuzzy_cats
    df["threshold_severity"] = thresh_scores
    demands = generate_zone_demands(df, resources)
    return df, resources, demands


@st.cache_resource
def execute_ga_cached(
    severities_tuple: tuple,
    resources_items: tuple,
    demands_df: pd.DataFrame,
    pop_tuple: tuple,
    pop_size: int,
    generations: int,
    seed: int,
) -> dict:
    """Run and cache the Genetic Algorithm resource allocator."""
    resources = dict(resources_items)
    t0 = time.perf_counter()
    ga = GAAllocator(
        severities=list(severities_tuple),
        resources=resources,
        demands=demands_df,
        population=list(pop_tuple),
        pop_size=pop_size,
        generations=generations,
        seed=seed,
    )
    result = ga.run()
    result["runtime_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return result


# ─── SIDEBAR & GLOBAL CONTROLS ───────────────────────────────────────────────
st.sidebar.title("⚙️ Simulation Controls")
st.sidebar.markdown("Configure synthetic disaster parameters and GA hyperparameters.")

n_zones = st.sidebar.slider("Number of Disaster Zones", min_value=5, max_value=20, value=10, step=1)
ga_pop_size = st.sidebar.slider("GA Population Size", min_value=20, max_value=120, value=60, step=10)
ga_generations = st.sidebar.slider("GA Generations", min_value=20, max_value=150, value=80, step=10)
seed_val = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

if st.sidebar.button("🔄 Regenerate Scenario", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("🚨 Hybrid Fuzzy–Genetic Algorithm Framework for Disaster Response")
st.markdown(
    "**Soft Computing Decision Support System:** Combines a **Mamdani Fuzzy Inference System** "
    "(Module A) to evaluate zone severity under uncertainty with a **Custom Genetic Algorithm** "
    "(Module B) to optimize multi-resource allocation under scarcity."
)

scenario_df, resources_dict, demands_df = load_scenario_bundle(seed=int(seed_val), n_zones=int(n_zones))

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Severity Assessment",
    "🧬 GA Resource Allocation",
    "⚖️ Comparison",
    "📊 Dataset & Analysis",
])

# ─── TAB 1: SEVERITY ASSESSMENT ──────────────────────────────────────────────
with tab1:
    st.subheader("Affected Disaster Zones Overview")
    st.dataframe(
        scenario_df[[
            "zone_id", "zone_name", "damage_level", "population_density",
            "accessibility", "casualty_count", "population", "fuzzy_severity", "severity_category"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("🔍 Interactive 'What-If' Zone Severity Simulator")
    col_sliders, col_result = st.columns([1.2, 1.8])

    with col_sliders:
        w_dmg = st.slider("Damage Level (0-10)", 0.0, 10.0, 7.5, 0.1)
        w_pop = st.slider("Population Density (0-10)", 0.0, 10.0, 6.5, 0.1)
        w_acc = st.slider("Accessibility Index (0-10, lower = harder)", 0.0, 10.0, 2.5, 0.1)
        w_cas = st.slider("Casualty Count Index (0-10)", 0.0, 10.0, 7.0, 0.1)

        whatif_res = assess_severity(w_dmg, w_pop, w_acc, w_cas)
        st.metric("Fuzzy Severity Score (0 - 100)", f"{whatif_res['score']:.2f}")
        cat = whatif_res["category"]
        if cat == "Low":
            st.success(f"Severity Category: **{cat}** (Routine Monitoring)")
        elif cat == "Moderate":
            st.warning(f"Severity Category: **{cat}** (Standard Dispatch)")
        else:
            st.error(f"Severity Category: **{cat}** (Priority Emergency Response)")

    with col_result:
        rules_df = pd.DataFrame(
            list(whatif_res["rule_strengths"].items()),
            columns=["Rule", "Firing Strength"],
        ).sort_values("Firing Strength", ascending=True)
        active_rules = rules_df[rules_df["Firing Strength"] > 0.0]
        if active_rules.empty:
            active_rules = rules_df.tail(5)

        fig_rules = px.bar(
            active_rules,
            x="Firing Strength",
            y="Rule",
            orientation="h",
            color="Firing Strength",
            color_continuous_scale="YlOrRd",
            range_x=[0, 1.05],
            title="Active Mamdani Fuzzy Rule Firing Strengths",
        )
        fig_rules.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_rules, use_container_width=True)

    st.divider()
    col_mf, col_surf = st.columns(2)
    with col_mf:
        st.subheader("Fuzzy Membership Functions")
        fig_mf = plot_membership_functions()
        st.pyplot(fig_mf)
    with col_surf:
        st.subheader("3D Fuzzy Control Surface")
        fig_surf = plot_severity_surface()
        st.plotly_chart(fig_surf, use_container_width=True)

# ─── TAB 2: GA RESOURCE ALLOCATION ───────────────────────────────────────────
with tab2:
    st.subheader("Genetic Algorithm Multi-Resource Optimization")
    with st.spinner("Running Genetic Algorithm Evolution..."):
        ga_res = execute_ga_cached(
            tuple(scenario_df["fuzzy_severity"].tolist()),
            tuple(sorted(resources_dict.items())),
            demands_df,
            tuple(scenario_df["population"].tolist()),
            int(ga_pop_size),
            int(ga_generations),
            int(seed_val),
        )

    ga_alloc_df = ga_res["allocation_df"]
    ga_eval = coverage_score(
        ga_alloc_df,
        scenario_df["fuzzy_severity"],
        demands_df,
        scenario_df["population"],
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Weighted Coverage Score", f"{ga_eval['total_score']:.2f}")
    m2.metric("Total Unmet Demand (Units)", f"{ga_eval['unmet_demand']}")
    m3.metric("Equity Gini Coefficient", f"{ga_eval['equity_gini']:.4f}")
    m4.metric("GA Runtime", f"{ga_res['runtime_ms']:.1f} ms")

    display_alloc = ga_alloc_df.copy()
    display_alloc.insert(1, "zone_name", scenario_df["zone_name"])
    display_alloc["fuzzy_severity"] = scenario_df["fuzzy_severity"]
    display_alloc["demand_amb / food / med"] = (
        demands_df["ambulances"].astype(str) + " / "
        + demands_df["food_kits"].astype(str) + " / "
        + demands_df["medical_teams"].astype(str)
    )
    display_alloc["coverage_%"] = display_alloc["zone_id"].map(ga_eval["per_zone_coverage"])

    st.dataframe(
        display_alloc.style.background_gradient(subset=["fuzzy_severity", "coverage_%"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

    c_conv, c_scat = st.columns(2)
    with c_conv:
        gen_df = pd.DataFrame({
            "Generation": list(range(len(ga_res["history"]))),
            "Best Fitness": ga_res["history"],
        })
        fig_conv = px.line(
            gen_df,
            x="Generation",
            y="Best Fitness",
            markers=True,
            title="GA Convergence Curve (Best Fitness vs. Generation)",
            color_discrete_sequence=["#d95f02"],
        )
        fig_conv.update_layout(xaxis_title="Generation", yaxis_title="Fitness Value")
        st.plotly_chart(fig_conv, use_container_width=True)

    with c_scat:
        fig_cov_scat = px.scatter(
            display_alloc,
            x="fuzzy_severity",
            y="coverage_%",
            size=scenario_df["population"],
            color="fuzzy_severity",
            hover_name="zone_name",
            color_continuous_scale="Turbo",
            title="Zone Fuzzy Severity vs. GA Allocated Coverage (%)",
            labels={"fuzzy_severity": "Fuzzy Severity Score", "coverage_%": "Resource Coverage (%)"},
        )
        st.plotly_chart(fig_cov_scat, use_container_width=True)

# ─── TAB 3: COMPARISON ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Benchmark Comparison: Hybrid Fuzzy+GA vs. Baselines")

    # Evaluate baselines and record execution times
    t0 = time.perf_counter()
    greedy_df = greedy_allocation(scenario_df["fuzzy_severity"], resources_dict, demands_df)
    greedy_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    greedy_eval = coverage_score(greedy_df, scenario_df["fuzzy_severity"], demands_df, scenario_df["population"])

    t0 = time.perf_counter()
    prop_df = proportional_allocation(scenario_df["fuzzy_severity"], resources_dict, demands_df)
    prop_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    prop_eval = coverage_score(prop_df, scenario_df["fuzzy_severity"], demands_df, scenario_df["population"])

    t0 = time.perf_counter()
    rand_df = random_allocation(scenario_df["fuzzy_severity"], resources_dict, demands_df, seed=int(seed_val))
    rand_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    rand_eval = coverage_score(rand_df, scenario_df["fuzzy_severity"], demands_df, scenario_df["population"])

    comp_table = pd.DataFrame([
        {
            "Method": "Fuzzy + GA (Proposed)",
            "total_score": ga_eval["total_score"],
            "unmet_demand": ga_eval["unmet_demand"],
            "equity_gini": ga_eval["equity_gini"],
            "runtime_ms": ga_res["runtime_ms"],
        },
        {
            "Method": "Greedy Allocation",
            "total_score": greedy_eval["total_score"],
            "unmet_demand": greedy_eval["unmet_demand"],
            "equity_gini": greedy_eval["equity_gini"],
            "runtime_ms": greedy_ms,
        },
        {
            "Method": "Proportional Allocation",
            "total_score": prop_eval["total_score"],
            "unmet_demand": prop_eval["unmet_demand"],
            "equity_gini": prop_eval["equity_gini"],
            "runtime_ms": prop_ms,
        },
        {
            "Method": "Random Allocation",
            "total_score": rand_eval["total_score"],
            "unmet_demand": rand_eval["unmet_demand"],
            "equity_gini": rand_eval["equity_gini"],
            "runtime_ms": rand_ms,
        },
    ])

    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_tot = px.bar(
            comp_table,
            x="Method",
            y="total_score",
            color="Method",
            text="total_score",
            title="Total Weighted Coverage Score by Allocation Method",
            labels={"total_score": "Total Weighted Score"},
        )
        st.plotly_chart(fig_tot, use_container_width=True)

    with col_b2:
        zone_comp_df = pd.DataFrame({
            "Zone": scenario_df["zone_id"],
            "Fuzzy + GA": [ga_eval["per_zone_coverage"][z] for z in scenario_df["zone_id"]],
            "Greedy Baseline": [greedy_eval["per_zone_coverage"][z] for z in scenario_df["zone_id"]],
        }).melt(id_vars="Zone", var_name="Method", value_name="Coverage (%)")

        fig_zcomp = px.bar(
            zone_comp_df,
            x="Zone",
            y="Coverage (%)",
            color="Method",
            barmode="group",
            title="Per-Zone Resource Coverage: Fuzzy+GA vs. Greedy",
        )
        st.plotly_chart(fig_zcomp, use_container_width=True)

    fig_agree = px.scatter(
        scenario_df,
        x="threshold_severity",
        y="fuzzy_severity",
        color="severity_category",
        hover_name="zone_name",
        title="Agreement Analysis: Mamdani Fuzzy Severity vs. Fixed-Threshold Baseline",
        labels={
            "threshold_severity": "Fixed-Weight Threshold Severity (0-100)",
            "fuzzy_severity": "Mamdani Fuzzy Severity (0-100)",
        },
    )
    fig_agree.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig_agree, use_container_width=True)

    st.info(
        "**Methodological Interpretation:** The **Hybrid Fuzzy + GA** framework achieves a high "
        "weighted coverage score while maintaining lower inequality (Gini) than purely myopic "
        "**Greedy Allocation**. While Greedy concentrates discrete resources (like ambulances and "
        "medical teams) into single top-severity zones regardless of population weight or multi-resource "
        "synergy, the GA jointly optimizes severity, population weight, and cross-zone balance. "
        "Additionally, the scatter plot shows that Mamdani fuzzy inference captures non-linear "
        "compound risks (e.g., high casualties + blocked access) that linear fixed weights underestimate."
    )

# ─── TAB 4: DATASET & ANALYSIS ───────────────────────────────────────────────
with tab4:
    st.subheader("Generated Disaster Scenario & Resource Demands")
    st.markdown(
        "The synthetic generator simulates correlated post-disaster dynamics: **casualty rates** "
        "increase with structural **damage** and **population density**, while road **accessibility** "
        "degrades inversely with damage. Resource demands are scaled to exceed supply (~1.85× scarcity)."
    )

    c_res, c_dem = st.columns([1, 2])
    with c_res:
        st.markdown("**Available Supply Pool**")
        st.dataframe(pd.DataFrame([resources_dict]), use_container_width=True, hide_index=True)
        st.markdown("**Total Demand Across Zones**")
        tot_dem = demands_df[["ambulances", "food_kits", "medical_teams"]].sum().to_dict()
        st.dataframe(pd.DataFrame([tot_dem]), use_container_width=True, hide_index=True)
    with c_dem:
        st.markdown("**Per-Zone Resource Demands Table**")
        st.dataframe(demands_df, use_container_width=True, hide_index=True)

    c_hist, c_corr = st.columns(2)
    with c_hist:
        fig_hist = px.histogram(
            scenario_df,
            x="fuzzy_severity",
            nbins=8,
            color="severity_category",
            title="Distribution of Fuzzy Severity Across Zones",
            labels={"fuzzy_severity": "Fuzzy Severity Score (0-100)"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_corr:
        corr_cols = ["damage_level", "population_density", "accessibility", "casualty_count", "fuzzy_severity"]
        corr_mat = scenario_df[corr_cols].corr().round(2)
        fig_heat = px.imshow(
            corr_mat,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap of Disaster Variables",
        )
        st.plotly_chart(fig_heat, use_container_width=True)