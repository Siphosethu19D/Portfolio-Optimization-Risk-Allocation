# Cardinality-Constrained Multi-Period Mean-Variance Portfolio Optimisation with Regime-Switching: Replication & Transaction Cost Extension

## PORA9X1 Group Project — University of Johannesburg

### Group Members

Siphosethu Limane, Keitumetse Motlhabase, Masixole Ntshangase, Joelle Behoor

---

### Project Overview

This project replicates and extends **Wang, Jin, Wu & Gao (2026)**, *"Cardinality constrained multi-period mean-variance portfolio optimization with regime-switching parameters"*, published in *Automatica* 183, 112669.

We implement the cardinality-constrained multi-period mean-variance (CMMV) portfolio optimisation model under Markov regime-switching dynamics, comparing it against two benchmarks:
- **CMV-static** — single-period buy-and-hold benchmark
- **CMMV-independent** — multi-period optimisation assuming i.i.d. returns (Theorem 11)

Our **extension** introduces proportional transaction costs and management fees, performing sensitivity analysis to assess how costs affect optimal portfolio allocation, turnover, and out-of-sample performance.

### Research Question

> How does the inclusion of proportional transaction costs affect optimal portfolio allocation, turnover, and out-of-sample performance in a cardinality-constrained multi-period mean–variance framework with regime-switching dynamics?

### Hypothesis

> Introducing transaction costs will reduce optimal portfolio cardinality and turnover, but the performance advantage of regime-switching strategies will persist relative to static benchmarks.

---

### File Structure

```
├── main.py                          # Entry point — runs full replication + extension
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── replication_paper.ipynb          # Interactive notebook (exploratory)
├── GROUP PORA ASS2.ipynb            # Original group assignment notebook
├── paper_text.txt                   # Extracted paper text for reference
├── replication_report.md            # Written replication report
├── replication_checklist.md         # Replication progress checklist
│
├── utils/                           # Shared utility modules
│   ├── __init__.py
│   ├── parameters.py                # Calibrated parameters (Costa & Araujo 2008)
│   ├── matrix_utils.py              # d_t(i), D_t(i), matrix regularisation (Eq. 8-9)
│   ├── simulation.py                # Markov regime-switching Monte Carlo simulation
│   └── risk_metrics.py              # SR, VaR, CVaR, MDD, turnover, Kupiec backtest
│
├── replication/                     # Pure replication (no costs)
│   ├── __init__.py
│   ├── solver.py                    # CCQO solver + theta recursion + 3 strategies
│   ├── run_replication.py           # Main replication runner (Table 1)
│   └── generate_figures.py          # Figures 1-3 (sample paths, frontier, SR vs T)
│
├── extension/                       # Transaction cost extension
│   ├── __init__.py
│   ├── cost_model.py                # TC, management fees, net wealth (post-opt)
│   ├── run_extension.py             # Extension runner (baseline + sensitivity)
│   └── generate_figures.py          # Figures 4-8 (costs, sensitivity, risk)
│
├── results/                         # Generated output (CSV, pickle)
│   ├── replication_results.csv      # SR table for q=1..10 (no costs)
│   ├── replication_comparison_table.csv  # Our values vs paper Table 1
│   ├── sample_path_data.pkl         # Representative sample path for Figure 1
│   ├── extension_baseline_results.csv    # Full risk metrics with baseline costs
│   ├── extension_sensitivity_results.csv # Sensitivity across TC levels
│   ├── performance_table_full.csv        # All metrics, q=1..10
│   └── sensitivity_table.csv             # SR summary across cost levels
│
├── figures/                         # Generated figures (PNG, 300 dpi)
│   ├── fig1_replication_sample_paths.png
│   ├── fig2_mv_efficient_frontier_q5.png
│   ├── fig3_sharpe_vs_horizon.png
│   ├── fig4_sharpe_vs_cardinality_with_costs.png
│   ├── fig5_sensitivity_sharpe_vs_q.png
│   ├── fig6_turnover_vs_q.png
│   ├── fig7_terminal_wealth_distribution.png
│   └── fig8_var_cvar_vs_q.png
│
└── data/                            # Raw data (if any)
```

### How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Gurobi (optional):**
   The CCQO solver uses Gurobi MIQP if available. Without Gurobi, the solver
   falls back to brute-force enumeration over all C(n, q) subsets — this works
   for n=10 but is slower.

   Free academic licence: https://www.gurobi.com/academia/academic-program-and-licenses/

3. **Run everything:**
   ```bash
   python main.py
   ```
   This runs:
   - Replication (Sharpe ratio table for q=1..10)
   - Replication figures (Figures 1-3)
   - Extension (transaction costs + sensitivity analysis)
   - Extension figures (Figures 4-8)

   All outputs are saved to `/results/` (CSV) and `/figures/` (PNG).

4. **Run individual components:**
   ```bash
   python replication/run_replication.py      # Replication only
   python replication/generate_figures.py     # Replication figures only
   python extension/run_extension.py          # Extension only
   python extension/generate_figures.py       # Extension figures only
   ```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Numerical computation |
| scipy | ≥1.10 | Statistical tests (Kupiec), chi-squared |
| pandas | ≥1.5 | Data tables and CSV I/O |
| matplotlib | ≥3.7 | Figure generation |
| seaborn | ≥0.12 | Plot styling |
| gurobipy | ≥10.0 | MIQP solver for CCQO (optional — brute-force fallback) |

### Methodology Notes

- **Shrinkage/regularisation for D matrix:** The second-moment matrix D_t(i) requires positive-definiteness for the CCQO solution. We apply ridge regularisation D_reg = D + εI with ε = 1e-6. See Cajas Ch. 3 (shrinkage estimation).

- **Cardinality constraints and real-world portfolio constraints:** The ||u||_0 ≤ q constraint limits the number of active assets, reflecting real-world portfolio management costs. See Cajas Ch. 9 (real-world constraints, cardinality, transaction costs).

- **Parameter sensitivity and robustness:** Our sensitivity analysis over transaction cost rates (α₁ ∈ {5bp, 10bp, 30bp, 50bp}) examines the robustness of the CMMV strategy. See Cajas Ch. 11 (robust optimisation, parameter sensitivity).

### Constraints & Limitations

1. **Myopic cost treatment** — Transaction costs are deducted post-optimisation, not integrated into the DP objective function. Results are a cost assessment, not a fully optimal cost-aware strategy. A dynamic cost-adjusted DP would require modifying the Bellman equation.

2. **Small sample / limited empirical data** — 8 quarters of calibration data, with heuristic regime classification rather than a fully calibrated HMM on a long time series. Covariance estimation is fragile with so few observations.

3. **D matrix regularisation** — The second-moment matrix required ridge regularisation (ε = 1e-6) to ensure positive definiteness. This introduces a small bias. See Cajas Ch. 3.

4. **Simulation scale** — The paper uses 10⁷ sample paths; we use 100,000. This may introduce sampling noise in terminal wealth statistics.

5. **Gurobi dependency** — MIQP sub-problems are solved optimally with Gurobi. The brute-force fallback is exact but slower for large n.

6. **Regime classification deviation** — The original paper calibrates transition probabilities from historical data; we use the reported values p(1,1) = 0.55, p(1,2) = 0.45, p(2,1) = 0.55, p(2,2) = 0.45 directly.

7. **No forward-looking cost integration** — The cardinality constraint is applied gross of costs. An optimal policy would consider the trade-off between rebalancing benefit and cost within the DP recursion.

8. **Calibration differences** — The relative ordering between the three strategies (CMMV > CMMV-independent > CMV-static) is correct and consistent with the paper. Absolute Sharpe ratio values differ from the paper due to calibration differences (parameter estimation, simulation scale, solver precision). This is expected and documented.

### References

- Wang, T., Jin, C., Wu, W., & Gao, J. (2026). Cardinality constrained multi-period mean-variance portfolio optimization with regime-switching parameters. *Automatica*, 183, 112669.
- Costa, O.L.V., & Araujo, M.V. (2008). A generalized multi-period mean-variance portfolio optimization with Markov switching parameters. *Automatica*, 44(10), 2487–2497.
- Cajas, D. — *Portfolio Optimisation: Theory and Practice* (Chapters 3, 9, 11).
- Li, D., & Ng, W.-L. (2000). Optimal dynamic portfolio selection: Multi-period mean-variance formulation. *Mathematical Finance*, 10(3), 387–406.
