"""
ml/generate_synthetic.py
------------------------
Generates the synthetic training dataset for the PPP-Risk classifier.

Design principles:
  - 15,000 balanced observations (50/50) for robust training
  - 34 features: 4 structural + 3 engineered + 5 composite risk indices
    + 21 binary behavioral flags (matching the full Joget form payload)
  - Realistic project lifecycle: early / mid / late phases drive
    different flag probabilities (governance risk peaks early,
    execution risk peaks mid, budget pressure peaks late)
  - Conditional correlations between related flags (e.g. delay →
    schedule_slip, cost_overruns → funding_risk)
  - Log-normal budget_gap calibrated to real PPP project statistics

Run:  python ml/generate_synthetic.py
Out:  ml/data/dataset_synthetic.csv
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N    = 15_000
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_synthetic.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

rng = np.random.default_rng(SEED)


# ── helpers ──────────────────────────────────────────────────────────────────
def bern(p): return rng.binomial(1, p, N).astype(float)
def cond(flag, p_yes, p_no): return np.where(flag == 1, rng.binomial(1, p_yes, N), rng.binomial(1, p_no, N)).astype(float)
def any_cond(*flags, p_yes, p_no): return cond(np.clip(sum(flags), 0, 1), p_yes, p_no)
def clamp(x, lo=0.0, hi=1.0): return np.clip(x, lo, hi)


# ── 1. Structural continuous features ────────────────────────────────────────
current_progress = rng.beta(1.5, 1.5, N)           # slightly bell-shaped, [0,1]
remaining_ratio  = 1.0 - current_progress

# Log-normal budget_gap: median ≈ 180k TND, realistic PPP range
budget_gap = np.exp(rng.normal(np.log(180_000), 0.70, N)).clip(8_000, 700_000)
nb_marches = rng.integers(1, 7, N).astype(float)   # 1–6 contracts


# ── 2. Project lifecycle phase (early < 0.35, mid 0.35–0.70, late > 0.70) ──
early = (current_progress < 0.35).astype(float)
mid   = ((current_progress >= 0.35) & (current_progress < 0.70)).astype(float)
late  = (current_progress >= 0.70).astype(float)


# ── 3. Execution flags (higher mid-to-late) ──────────────────────────────────
# delay probability increases with progress
p_delay = clamp(0.15 + 0.45 * current_progress + rng.normal(0, 0.05, N))
delay = (rng.uniform(0, 1, N) < p_delay).astype(float)

schedule_slip = cond(delay,        p_yes=0.62, p_no=0.08)
pacing        = cond(delay,        p_yes=0.58, p_no=0.16)
pressure      = cond(schedule_slip, p_yes=0.68, p_no=0.20)


# ── 4. Budget flags (higher late; cost_overruns / funding_risk correlated) ───
# cost_overruns peaks in late phase
p_cost_overruns = clamp(0.15 + 0.35 * late + 0.10 * mid + rng.normal(0, 0.05, N))
cost_overruns = (rng.uniform(0, 1, N) < p_cost_overruns).astype(float)

# funding_risk: tight budget OR user-reported concern
_tight       = (budget_gap < 80_000).astype(float)
funding_risk = np.minimum(1, _tight + bern(0.10)).astype(float)

budget_revisions = cond(cost_overruns, p_yes=0.60, p_no=0.14)
margin_pressure  = any_cond(cost_overruns, funding_risk, p_yes=0.65, p_no=0.16)


# ── 5. Operational flags (higher mid; contractor / resource correlated) ───────
errors            = clamp(bern(0.18) + 0.12 * mid)
errors            = (rng.uniform(0, 1, N) < errors).astype(float)
contractor_issues = cond(errors,            p_yes=0.58, p_no=0.16)
resource_shortage = bern(0.22)
coordination      = any_cond(contractor_issues, resource_shortage, p_yes=0.62, p_no=0.13)


# ── 6. External flags (independent but correlated pairs) ─────────────────────
external_risk   = bern(0.24)
supplier_delays = cond(external_risk, p_yes=0.58, p_no=0.13)
regulatory      = bern(0.18)
client_changes  = bern(0.22)
dependency      = any_cond(regulatory, external_risk, p_yes=0.52, p_no=0.12)


# ── 7. Governance flags (higher early: poor setup; also late: crisis mgmt) ───
# decision_delay peaks early (setup phase) and late (pressure)
p_decision_delay = clamp(0.25 * early + 0.18 * mid + 0.30 * late + rng.normal(0, 0.04, N))
decision_delay   = (rng.uniform(0, 1, N) < p_decision_delay).astype(float)

reporting      = cond(decision_delay,  p_yes=0.58, p_no=0.14)
risk_tracking  = cond(decision_delay,  p_yes=0.52, p_no=0.16)
escalation     = any_cond(pressure, decision_delay, p_yes=0.58, p_no=0.08)


# ── 8. Engineered features ────────────────────────────────────────────────────
_rem_safe        = np.maximum(remaining_ratio, 0.01)
progress_pressure = current_progress / (_rem_safe + 0.01)
cost_pressure     = budget_gap / (_rem_safe + 1.0)
budget_risk       = ((budget_gap < 150_000).astype(float) + funding_risk) / 2.0


# ── 9. Composite risk indices (mean of each group) ───────────────────────────
execution_index   = (delay + schedule_slip + pacing + pressure) / 4.0
budget_index      = (cost_overruns + budget_revisions + funding_risk + margin_pressure) / 4.0
operational_index = (errors + contractor_issues + resource_shortage + coordination) / 4.0
external_index    = (client_changes + supplier_delays + regulatory + external_risk + dependency) / 5.0
governance_index  = (reporting + decision_delay + risk_tracking + escalation) / 4.0
total_risk_density = (
    delay + schedule_slip + pacing + pressure +
    cost_overruns + budget_revisions + funding_risk + margin_pressure +
    errors + contractor_issues + resource_shortage + coordination +
    client_changes + supplier_delays + regulatory + external_risk + dependency +
    reporting + decision_delay + risk_tracking + escalation
) / 21.0


# ── 10. Risk score — group-weighted, lifecycle-aware ─────────────────────────
budget_norm     = (budget_gap.max() - budget_gap) / (budget_gap.max() - budget_gap.min() + 1e-8)
nb_marches_norm = (nb_marches - 1) / 5.0

risk_score = (
    0.18 * current_progress    +   # structural: advanced = budget consumed
    0.12 * budget_norm         +   # structural: tight budget = higher risk
    0.02 * nb_marches_norm     +   # structural: complexity
    0.14 * execution_index     +   # execution problems
    0.14 * budget_index        +   # budget problems
    0.12 * operational_index   +   # operational problems
    0.12 * external_index      +   # external problems
    0.08 * governance_index    +   # governance problems
    0.08 * total_risk_density      # overall signal density
)

# 3.5% noise → median threshold guarantees 50/50
risk_score += rng.normal(0, 0.035, N)
target = (risk_score >= np.median(risk_score)).astype(int)


# ── 11. Assemble DataFrame ────────────────────────────────────────────────────
df = pd.DataFrame({
    # Structural
    "current_progress": current_progress.round(4),
    "remaining_ratio":  remaining_ratio.round(4),
    "budget_gap":       budget_gap.round(2),
    "nb_marches":       nb_marches.astype(int),
    # Engineered
    "progress_pressure":  progress_pressure.round(4),
    "cost_pressure":      cost_pressure.round(4),
    "budget_risk":        budget_risk.round(4),
    # Composite indices
    "execution_index":    execution_index.round(4),
    "budget_index":       budget_index.round(4),
    "operational_index":  operational_index.round(4),
    "external_index":     external_index.round(4),
    "governance_index":   governance_index.round(4),
    "total_risk_density": total_risk_density.round(4),
    # Execution
    "delay":           delay.astype(int),
    "schedule_slip":   schedule_slip.astype(int),
    "pacing":          pacing.astype(int),
    "pressure":        pressure.astype(int),
    # Budget
    "cost_overruns":   cost_overruns.astype(int),
    "budget_revisions":budget_revisions.astype(int),
    "funding_risk":    funding_risk.astype(int),
    "margin_pressure": margin_pressure.astype(int),
    # Operational
    "errors":            errors.astype(int),
    "contractor_issues": contractor_issues.astype(int),
    "resource_shortage": resource_shortage.astype(int),
    "coordination":      coordination.astype(int),
    # External
    "client_changes":  client_changes.astype(int),
    "supplier_delays": supplier_delays.astype(int),
    "regulatory":      regulatory.astype(int),
    "external_risk":   external_risk.astype(int),
    "dependency":      dependency.astype(int),
    # Governance
    "reporting":       reporting.astype(int),
    "decision_delay":  decision_delay.astype(int),
    "risk_tracking":   risk_tracking.astype(int),
    "escalation":      escalation.astype(int),
    # Target
    "risque_depassement": target,
})

df.to_csv(OUT, index=False)

# ── 12. Summary ───────────────────────────────────────────────────────────────
vc  = df["risque_depassement"].value_counts()
pct = df["risque_depassement"].value_counts(normalize=True)
print(f"Generated {len(df):,} rows  ->  {OUT}")
print(f"  Classe 0 (sans risque) : {vc.get(0,0):,} ({pct.get(0,0)*100:.1f}%)")
print(f"  Classe 1 (à risque)    : {vc.get(1,0):,} ({pct.get(1,0)*100:.1f}%)")
print(f"\nFeatures: {len(df.columns)-1} (4 structurel + 3 dérivé + 6 composites + 21 binaires)")
print(f"\nSignaux binaires — delta de prévalence (classe 1 vs classe 0):")
flags = ["delay","schedule_slip","pacing","pressure",
         "cost_overruns","budget_revisions","funding_risk","margin_pressure",
         "errors","contractor_issues","resource_shortage","coordination",
         "client_changes","supplier_delays","regulatory","external_risk","dependency",
         "reporting","decision_delay","risk_tracking","escalation"]
for f in flags:
    p0 = df[df["risque_depassement"]==0][f].mean()
    p1 = df[df["risque_depassement"]==1][f].mean()
    print(f"  {f:<22}  cl0={p0:.2f}  cl1={p1:.2f}  d={p1-p0:+.2f}")
