# filename: README.md
# Hybrid Fuzzy–Genetic Algorithm Framework for Disaster Response

## 1. Overview
Emergency response following a natural disaster requires rapid, high-stakes decision-making under uncertainty and resource scarcity. Field reports on structural damage, road accessibility, and casualties are inherently imprecise, making crisp threshold triggers brittle. Furthermore, distributing limited resources (ambulances, food kits, and medical teams) across competing zones is a constrained combinatorial optimization problem. This project combines a **Mamdani Fuzzy Inference System (FIS)** for nuanced severity evaluation with a **Custom Genetic Algorithm (GA)** for optimal resource dispatch, wrapped in an interactive Streamlit analytics application.

## 2. Guidelines Mapping (7-Point Table)

| # | Course Requirement | Implementation File(s) | Verification / Details |
|---|---|---|---|
| 1 | Module A: Mamdani Fuzzy Inference | `fuzzy_severity.py` | 4 antecedents (`0-10`), 1 consequent (`0-100`), 22 rules, centroid defuzzification via `scikit-fuzzy`. |
| 2 | Module B: Custom Genetic Algorithm | `ga_allocator.py` | Hand-rolled integer GA (`N×3` genes), tournament selection, uniform crossover, random-reset mutation, supply repair. |
| 3 | Correlated Synthetic Scenario & Scarcity | `data_generator.py` | Seeded NumPy RNG (`default_rng`); inverse damage-accessibility & positive damage-casualty correlations; ~1.85× demand scarcity. |
| 4 | Deterministic & Heuristic Baselines | `baselines.py` | Fixed-weight linear severity (`[0.35, 0.2, 0.15, 0.3]`), Greedy, Proportional, and Random allocators. |
| 5 | Multi-Metric Evaluation | `baselines.py` | `coverage_score()` computes `total_score`, `unmet_demand`, `equity_gini`, and `per_zone_coverage`. |
| 6 | 4-Tab Interactive Streamlit Dashboard | `app.py` | Tabs for Severity Assessment, GA Resource Allocation, Baseline Comparison, and Dataset Analysis. |
| 7 | Reproducibility & Cloud Readiness | `requirements.txt`, `DEPLOY.md` | Deterministic seeding throughout; zero external API dependencies; 3-step Streamlit Cloud deployment. |

## 3. Architecture Diagram

```text
+-------------------------------------------------------------------------+
|                     SYNTHETIC SCENARIO GENERATOR                        |
|   data_generator.py -> Correlated Zones (N) + Scarce Supply + Demands   |
+------------------------------------+------------------------------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
+--------------------------------+      +---------------------------------+
|   MODULE A: MAMDANI FUZZY FIS  |      |      BASELINE THRESHOLDING      |
|        fuzzy_severity.py       |      |           baselines.py          |
| - 4 Antecedents (Low/Med/High) |      | - Linear Weighted Sum           |
| - 22 Domain IF-THEN Rules      |      |   [0.35, 0.20, 0.15, 0.30]      |
| - Centroid Defuzzification     |      +----------------+----------------+
+----------------+---------------+                       |
                 |  Fuzzy Severity Scores (0-100)        |
                 +-------------------+-------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               MODULE B: GENETIC ALGORITHM vs. BASELINES                 |
|  - GAAllocator (ga_allocator.py): Tournament + Uniform Xover + Repair   |
|  - Baselines (baselines.py): Greedy, Proportional, Random               |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   STREAMLIT VISUALIZATION DASHBOARD                     |
|  app.py -> Tab 1: FIS | Tab 2: GA | Tab 3: Comparison | Tab 4: Data     |
+-------------------------------------------------------------------------+