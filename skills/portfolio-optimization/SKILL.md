---
name: portfolio-optimization
description: Use when a user asks to build, optimize, backtest, rebalance, or analyze a stock portfolio with Mean-CVaR, Mean-Variance/SOCP variance caps, efficient frontiers, scenario generation, or NVIDIA cuOpt.
license: Apache-2.0
metadata:
  author: Jake Goldberg <jgoldberg@nvidia.com>
  tags:
    - portfolio-optimization
    - cvar
    - cuopt
    - quantitative-finance
    - gpu
---

# Portfolio Optimization with NVIDIA cuOpt

<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Build and analyze quantitative portfolios with NVIDIA-accelerated Mean-CVaR and Mean-Variance optimization. Use the `portfolio_optimization` package to compute returns, generate KDE scenarios for CVaR, solve variance-cap Markowitz allocations as SOCP/QCQP problems with the cuOpt GPU solver, trace an efficient frontier, backtest portfolios, and run rebalancing workflows from price data.

## When to Use

Use this skill when the task is to:

- Build or optimize a Mean-CVaR portfolio from stock prices.
- Allocate weights across tickers while controlling downside CVaR risk.
- Solve Mean-Variance or Markowitz allocations with a hard variance or volatility cap using cuOpt SOCP/QCQP support.
- Plot or inspect an efficient frontier for a portfolio universe.
- Produce a weights-by-risk-aversion table.
- Backtest an optimized portfolio against benchmarks.
- Rebalance a portfolio on a schedule or drift trigger.
- Run workflows on an S&P 500, S&P 100, Dow 30, or user-supplied price dataset.

Common trigger phrases include "optimize my portfolio", "build a CVaR portfolio", "use cuOpt to optimize these tickers", "solve with cuOpt", "plot the efficient frontier", "show weights by risk aversion", "backtest this allocation", "rebalance monthly", "analyze my holdings with CVaR", "compare allocations", "reduce downside risk", "construct an allocation", "assess allocation options", "stress-test my holdings", "evaluate downside-risk exposure", "review my holdings under weight caps", "compare benchmark portfolios", "simulate CVaR scenarios", "screen portfolio risk", "optimize holdings under constraints", "solve a variance-cap portfolio", "use SOCP", "set a volatility cap", and "find a lower-risk allocation".

Do not use it for generic finance summaries, price forecasting, neural-network training, vehicle routing, or non-portfolio optimization.

## Prerequisites

- Python environment with the installed `portfolio_optimization` package.
- NVIDIA GPU runtime with cuOpt and cuML installed. Mean-Variance SOCP workflows require a cuOpt build with QCQP/SOCP support, such as the 26.06 line or newer.
- CUDA extra matching the host and workflow: `uv sync --extra cuda12` for full cuOpt/cuML 26.06 on CUDA 12, `uv sync --extra cuda13` for the current full CUDA 13 stack, or `uv sync --extra cuda13-socp` for CUDA 13 SOCP-only validation with cuOpt 26.06.
- `cvxpy` exposing `cp.CUOPT`.
- Network access on first run if the default price CSV must be downloaded.

## Setup

This skill drives the installed `portfolio_optimization` package. A ready environment can come from the Brev launchable or from the `NVIDIA-AI-Blueprints/portfolio-optimization` repository after installing the matching CUDA extra.

In packaged agent/eval sandboxes, `portfolio_optimization` may be available through `PYTHONPATH` rather than as a separately published wheel. Verify the local package with `python -c "import portfolio_optimization"` before declaring it missing. Do not `pip install portfolio_optimization`; do not reimplement the example workflows from scratch, and do not replace the package APIs with generic pandas/scipy/cvxpy portfolio code.

For concrete implementation details, use `references/workflows/agent_recipes.md` as the source of truth. It contains exact working shapes for loading prices, preparing returns, solving with cuOpt, building a 25-point frontier, backtesting against equal weight, and calling the rebalancer.

The default dataset is `data/stock_data/sp500.csv`. It is gitignored. Before a first-run download, tell the user this fetches public market data through the package's yfinance data helper and ask them to confirm:

```python
import cvxpy as cp
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.utils import download_data

download_data("data/stock_data", datasets=["sp500"])
CVAR_SOLVER_SETTINGS = {"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"}
cvar_params = CvarParameters(
    w_min=0.0, w_max=1.0,
    c_min=0.0, c_max=0.0,
    risk_aversion=1.0, confidence=0.95,
)
```

## Instructions

Briefly state the defaults being applied before execution, then use these guardrails:

1. Load `data/stock_data/sp500.csv`; if it is missing, ask before downloading `sp500` with `portfolio_optimization.utils.download_data`. Do not glob, substitute, or fabricate price data.
2. Validate user CSVs before solving: require a date-like index or first date column, numeric ticker columns, at least 60 rows after date filtering, and at least one requested ticker. If the user gives start/end dates, slice the price DataFrame before returns computation and report the retained date range. Filter tickers on the price DataFrame before returns are computed. `regime_dict` does not take a ticker field.
3. Compute LOG returns with `utils.calculate_returns(...)`.
4. For Mean-CVaR tasks, generate scenarios with `cvar_utils.generate_cvar_data(...)`, KDE, and `KDESettings(device="GPU")`. For Mean-Variance SOCP variance-cap tasks, do not generate CVaR scenarios; use the `returns_dict` directly after LOG return computation.
5. For ordinary Mean-CVaR portfolio requests, define `CvarParameters` with explicit `w_min` and `w_max`, and set `c_min=0.0` and `c_max=0.0` so the result is fully invested instead of 100 percent cash.
6. For variance-cap, volatility-cap, Markowitz, SOCP, or QCQP requests, define `MeanVarianceParameters` with `var_limit` set to a positive variance bound, `c_min=0.0`, `c_max=0.0`, and `L_tar=1.0` for long-only fully invested allocations. If the user gives a volatility cap, square it before assigning `var_limit`.
7. Build `cvar_optimizer.CVaR(returns_dict, cvar_params)` for Mean-CVaR tasks. Build `mean_variance_optimizer.MeanVariance(returns_dict, mean_variance_params, api_settings=ApiSettings(api="cuopt_python"))` for direct cuOpt Mean-Variance SOCP tasks.
8. Solve with NVIDIA cuOpt only. For CVaR, verify `hasattr(cp, "CUOPT")` and `str(cp.CUOPT) in {str(s) for s in cp.installed_solvers()}`, then pass `CVAR_SOLVER_SETTINGS` to every single-shot solve or looped frontier solve. For direct Mean-Variance SOCP, verify the `cuopt` Python package is importable and call the optimizer with `api="cuopt_python"`; cuOpt auto-selects the barrier method for quadratic constraints. Never fall back to CLARABEL, SCS, ECOS, or another CPU solver. If cuOpt is absent, finish validation/setup and report that the GPU/cuOpt runtime is missing instead of fabricating a CPU result.
9. For custom constraints, map user requests to the appropriate parameter model: CVaR risk controls to `CvarParameters`, variance or volatility caps to `MeanVarianceParameters.var_limit`, weight caps to `w_min`/`w_max`, risk appetite to `risk_aversion`, confidence level to `confidence`, and cash allowance to `c_max`. Treat cardinality plus SOCP as unsupported unless the package exposes explicit mixed-integer conic support.
10. If the user omits a benchmark for backtesting, use an equal-weight portfolio over the same tickers. If the user omits a constraint, keep the defaults table values and briefly restate consequential assumptions before solving.
11. Deliver weights sorted by allocation, cash weight, expected return, solver label (`cuOpt GPU`), and the risk metric used: CVaR for Mean-CVaR or realized variance plus `var_limit` for SOCP. Include any requested frontier figure, weights table, backtest metrics, or rebalancing schedule. For tables, include tickers as columns or rows with decimal weights and percentages; for plots, preserve the figure returned by the package instead of redrawing from scratch.
12. For report-grade answers, include evidence that the requested workflow actually ran. For an efficient frontier, state `len(results_df)` and use the requested `ra_num` (25 unless the user specifies otherwise). For a variance-cap SOCP solve, report `result_row["solver"]`, realized variance, the requested `var_limit`, and confirm realized variance is at or below the cap. For a weights table, expand `results_df["weights"]` into ticker columns and include `cash` plus `risk_aversion`. For a backtest, include `mean portfolio return`, `sharpe`, `sortino`, and `max drawdown` for both optimized and benchmark portfolios. For rebalancing, include `results_dataframe`, `re_optimize_dates`, and the tail of `cumulative_portfolio_value`.

## Canonical Workflow Skeleton

Start applicable portfolio optimization tasks from this shape and adapt only the requested output. For complete copyable functions, read `references/workflows/agent_recipes.md` before writing custom code.

### Mean-CVaR workflow

```python
import cvxpy as cp
import pandas as pd

from portfolio_optimization import backtest, cvar_optimizer, cvar_utils, rebalance, utils
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.settings import KDESettings, ReturnsComputeSettings, ScenarioGenerationSettings

if not hasattr(cp, "CUOPT") or str(cp.CUOPT) not in {str(s) for s in cp.installed_solvers()}:
    raise RuntimeError("cuOpt GPU solver is required; do not substitute a CPU solver.")

CVAR_SOLVER_SETTINGS = {"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"}

prices = utils.get_input_data("data/stock_data/sp500.csv")
returns_dict = utils.calculate_returns(
    prices,
    regime_dict=None,
    returns_compute_settings=ReturnsComputeSettings(return_type="LOG"),
)
returns_dict = cvar_utils.generate_cvar_data(
    returns_dict,
    ScenarioGenerationSettings(
        fit_type="kde",
        kde_settings=KDESettings(device="GPU"),
    ),
)
cvar_params = CvarParameters(
    w_min=0.0,
    w_max=1.0,
    c_min=0.0,
    c_max=0.0,
    risk_aversion=1.0,
    confidence=0.95,
)
optimizer = cvar_optimizer.CVaR(returns_dict, cvar_params)
result, optimal_portfolio = optimizer.solve_optimization_problem(
    solver_settings=CVAR_SOLVER_SETTINGS,
    print_results=False,
)
```

### Mean-Variance SOCP workflow

```python
import importlib.util
import numpy as np

from portfolio_optimization import mean_variance_optimizer, utils
from portfolio_optimization.mean_variance_parameters import MeanVarianceParameters
from portfolio_optimization.settings import ApiSettings, ReturnsComputeSettings

if importlib.util.find_spec("cuopt") is None:
    raise RuntimeError("cuOpt Python API is required; do not substitute a CPU solver.")

prices = utils.get_input_data("data/stock_data/sp500.csv")
returns_dict = utils.calculate_returns(
    prices,
    regime_dict=None,
    returns_compute_settings=ReturnsComputeSettings(return_type="LOG"),
)
weights = np.ones(len(returns_dict["tickers"])) / len(returns_dict["tickers"])
var_limit = float(weights @ returns_dict["covariance"] @ weights) * 1.05
mean_variance_params = MeanVarianceParameters(
    w_min=0.0,
    w_max=1.0,
    c_min=0.0,
    c_max=0.0,
    L_tar=1.0,
    var_limit=var_limit,
)
optimizer = mean_variance_optimizer.MeanVariance(
    returns_dict,
    mean_variance_params,
    api_settings=ApiSettings(api="cuopt_python"),
)
result, optimal_portfolio = optimizer.solve_optimization_problem(print_results=False)
realized_variance = float(
    optimal_portfolio.weights @ returns_dict["covariance"] @ optimal_portfolio.weights
)
```

For an efficient frontier or weights table, call:

```python
results_df, fig, ax = cvar_utils.create_efficient_frontier(
    returns_dict,
    cvar_params,
    CVAR_SOLVER_SETTINGS,
    ra_num=25,
    show_plot=False,
    show_discretized_portfolios=False,
    benchmark_portfolios=False,
    print_portfolio_results=False,
)
weights_table = pd.DataFrame(results_df["weights"].tolist(), index=results_df.index)
```

For a benchmark backtest, wrap the solved allocation in `Portfolio(name="cuOpt Optimal", tickers=returns_dict["tickers"], weights=optimal_portfolio.weights, cash=optimal_portfolio.cash)`, create an equal-weight `Portfolio` over the same `returns_dict["tickers"]`, then use `backtest.portfolio_backtester(..., test_method="historical").backtest_against_benchmarks(...)`. The backtester returns `(backtest_results, ax)`.

For monthly rebalancing, write the price DataFrame to a CSV path first. Instantiate `rebalance.rebalance_portfolio(dataset_directory=<csv_path>, ...)` with `re_optimize_criteria={"type": "drift_from_optimal", "threshold": 0, "norm": 1}` and call `re_optimize(transaction_cost_factor=..., plot_title="Monthly Rebalancing")`. The rebalancer returns `(results_dataframe, re_optimize_dates, cumulative_portfolio_value)`.

## Data and Defaults

| Setting | Default |
|---|---|
| Dataset | `data/stock_data/sp500.csv` |
| Date range | Full available range |
| Portfolio type | Long-only |
| Max weight | None unless specified |
| Risk aversion | `1.0` |
| Confidence | `0.95` |
| Scenario method | KDE on GPU |
| Solver | CVaR: cuOpt GPU with PDLP; Mean-Variance SOCP: direct cuOpt Python API with barrier auto-selected |
| Rebalancing | None unless requested |

The default S&P 500 file is a historical snapshot and can omit current constituents. User-supplied CSVs should be date-indexed price tables with ticker columns, compatible with `utils.get_input_data`. If requested tickers are absent, drop them, report the omissions, and continue with available columns unless the user explicitly asks you to fetch other data.

## Key APIs

Use the package APIs instead of reimplementing portfolio math or simulation loops. `portfolio_optimization` helpers return flat objects: `returns_dict` has keys such as `returns`, `mean`, `covariance`, and `tickers`; do not index it as `returns_dict["regime_1"]`. `solve_optimization_problem(...)` returns `(result_row, portfolio)`, not a nested result dictionary.

- Returns: `utils.calculate_returns(input_dataset, regime_dict, returns_compute_settings)`.
- Regime filter: `regime_dict` is `None` or `{"name": "...", "range": ("YYYY-MM-DD", "YYYY-MM-DD")}`; it is not keyed by regime name and does not contain tickers.
- Scenarios: `cvar_utils.generate_cvar_data(returns_dict, scenario_generation_settings)` for Mean-CVaR only.
- CVaR optimizer: `cvar_optimizer.CVaR(returns_dict, cvar_params)`.
- Mean-Variance SOCP optimizer: `mean_variance_optimizer.MeanVariance(returns_dict, mean_variance_params, api_settings=ApiSettings(api="cuopt_python"))`.
- CVaR solve: `result_row, portfolio = cvar_problem.solve_optimization_problem(solver_settings=CVAR_SOLVER_SETTINGS, print_results=False)`.
- SOCP solve: `result_row, portfolio = mean_variance_problem.solve_optimization_problem(print_results=False)`.
- Efficient frontier: `cvar_utils.create_efficient_frontier(returns_dict, cvar_params, solver_settings=CVAR_SOLVER_SETTINGS, ra_num=25)`. The returned `results_df` includes metrics, a `weights` dict column, and `cash`.
- Portfolio: `Portfolio(name="", tickers=None, weights=None, cash=0.0, time_range=None)`; pass tickers and a flat array-like `weights` aligned to those tickers.
- Backtest: create `portfolio.Portfolio` objects for the optimized allocation and each benchmark; for an equal-weight benchmark, use weights of `1 / len(tickers)` and `cash=0.0`, then call `backtest.portfolio_backtester(test_portfolio, returns_dict, risk_free_rate=0.0, test_method="historical", benchmark_portfolios=[...]).backtest_against_benchmarks(...)`.
- Rebalance: `rebalance.rebalance_portfolio(...)` requires `dataset_directory` to be a CSV path, not a DataFrame. Call `re_optimize(...)`; it returns `(results_dataframe, re_optimize_dates, cumulative_portfolio_value)`.
- Settings models: `ReturnsComputeSettings`, `ScenarioGenerationSettings`, `KDESettings`, `ApiSettings`, `CvarParameters`, and `MeanVarianceParameters`.

## Examples

- "Build the optimal portfolio from the S&P 500": load prices, compute LOG returns, generate GPU KDE scenarios, set long-only fully invested `CvarParameters`, solve with cuOpt, and report diversified weights plus return/CVaR.
- "Solve a variance-cap portfolio with SOCP": load prices, compute LOG returns, set `MeanVarianceParameters(var_limit=...)`, solve with direct `api="cuopt_python"`, and report expected return, realized variance, `var_limit`, and weights.
- "Plot the efficient frontier": call `create_efficient_frontier(...)`, return `results_df`, and show or save the figure as requested.
- "Give me weights by risk aversion": expand `results_df["weights"]` into a per-asset table.
- "Backtest against equal weight": build the optimized and equal-weight `Portfolio` objects, then use the package backtester and report Sharpe, Sortino, and max drawdown.
- "Backtest monthly rebalancing": configure `rebalance_portfolio` with the drift trigger above and run `re_optimize(transaction_cost_factor=...)`.

## Limitations

- Requires an NVIDIA GPU with cuOpt and cuML; CPU solvers are intentionally disallowed.
- Mean-Variance SOCP variance caps require cuOpt QCQP/SOCP support. Use the 26.06 line or newer when installing CUDA extras.
- `cuda13-socp` intentionally installs cuOpt without cuML because `cuml-cu13` 26.06 is not published yet; use it for direct SOCP/QCQP validation, not GPU KDE CVaR workflows.
- Cardinality plus SOCP is treated as unsupported unless the package exposes explicit mixed-integer conic support.
- CPU-only eval containers can still validate routing, data handling, and reporting behavior, but they cannot produce a valid cuOpt solve. In that case, report the missing GPU/cuOpt runtime explicitly.
- Default price data is a historical snapshot and may omit current constituents.
- First-run dataset download depends on network access unless the user supplies a CSV.

## Troubleshooting

- Missing default CSV or `FileNotFoundError`: explain that the package will fetch public market data with `download_data("data/stock_data", datasets=["sp500"])`; run it only after user confirmation.
- `SolverError` or missing `cp.CUOPT`: install the CUDA extra matching the host and verify with `python -c "import cvxpy as cp; print(hasattr(cp, 'CUOPT'), cp.installed_solvers())"`.
- `ImportError` for `cuml` or GPU KDE failures: confirm cuML is present with `python -c "import cuml"` and keep `KDESettings(device="GPU")`. If using `cuda13-socp`, this is expected for CVaR/KDE; switch to `cuda12` or `cuda13` for cuML workflows.
- SOCP setup fails before solving: verify the installed `cuopt` package is on the 26.06 line or newer and that `MeanVarianceParameters.var_limit` is positive.
- Ordinary optimization returns all cash: set `c_max=0.0` in `CvarParameters`.
- Solver reports infeasible or no solution: check for contradictory bounds, too few tickers for the requested caps/cardinality, or a date filter that leaves too little data; report the smallest constraint change that would make the request feasible.
- Requested tickers are absent from the default CSV: report them and proceed with the remaining requested tickers.
- User CSV fails validation: ask for a date-indexed price table or a CSV whose first column is dates and remaining columns are numeric ticker prices; mention the minimum 60-row post-filter requirement.
