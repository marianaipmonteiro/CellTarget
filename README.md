# CellTarget
Package to infer cellular objectives with convex optimisation

## 1) Overview
This repository implements **inverse Flux Balance Analysis (iFBA)** with a differentiable convex optimization layer (CVXPYLayer). The goal is to infer **objective coefficients** \(C\) such that the solution of a forward FBA problem reproduces measured fluxes.

- **Full observability**: all fluxes are treated as measured.
- **Partial observability**: only a subset of fluxes is matched; the layer objective depends only on that subset.

Both scripts support multiple **objective bases** (`norm`, `sum_squares`, `logsumexp`, `inverse_log`) and use **multi-start** with **Bayesian optimization** for initialization, followed by gradient-based refinement (LBFGS/SGD).

## 2) Inputs 
###Full observability

- Inputs are already provided. 

**Run**
```bash
python full_observability.py
```

###Partial observability

- **Models (JSON)**  
  - `inputs/CHOmpact_cobrapy_original.json` (for `early` / `stat`)  
  - `inputs/CHOmpact_cobrapy_original_nonprod.json` (for `AWZ11expo` / `AWZ11stat`)
- **Bounds & medians (Excel)**  
  - `inputs/C13_data.xlsx`, sheet `c13`, with rows `<condition>_LB`, `<condition>_UB`, `<condition>_MEDIAN` and columns named by reaction IDs.
- **Condition** (`--condition`): one of `early`, `stat`, `AWZ11expo`, `AWZ11stat`.

**Run**
```bash
python partial_observability_cho.py --condition <condition> --objective <objective> --output_dir <output_dir>
```


## 3) Outputs
Results are written to `--output_dir/<condition>/<mode>/`:

- `run_XXX/` per start:
  - `coefficients_<mode>.pdf`
  - `flux_comparison_<mode>.pdf`
  - `loss_history_<mode>.pdf` and `.csv`
  - `grad_per_dimension_<mode>.png` and `.csv`
  - `flux_summary.txt`
- `final/` for the best run:
  - `coefficients_<mode>.pdf`, `flux_comparison_<mode>.pdf`
  - `flux_summary.txt`

## 4) Benchmark

**Purpose.** Provide a convex benchmark for CellTarget by inferring an **objective vector** \(c\) that makes a forward FBA solution consistent with measured fluxes, via **KKT-based constraints** with convex surrogate complementarity.

**Inputs**
- COBRA model JSON (e.g., `inputs/CHOmpact_cobrapy_original.json`).
- C13 bounds/medians Excel: `inputs/C13_data.xlsx`, sheet `c13`, with rows
  `<condition>_LB`, `<condition>_UB`, `<condition>_MEDIAN`.
- Config in `__main__`:  
  `conditions = ['early','stat']`, objective flux IDs (e.g., `['F142','F143','F105','F107','F112','F114']`), and \(S_v\) (e.g., `0.001`).

**Outputs**
- CSV at `ThesisResults_V6/C13/ObjFind/prod_objfind_Sv{S_v}.csv` containing:
  - `loss` (data-fit + penalty value),
  - inferred `c_opt`,
  - measured fluxes (from medians),
  - forward FBA fluxes `v_fba_*`,
  - `status` (solver status).

**Run**
```bash
python objfind_benchmark.py
```

## 5) Results

All the results are stored in the `Publication_Results/C13/` directory.
The FullObservability results can be replicated when running:

```bash
python full_observability.py
```

The following file can be run to replicate the results of the Partial Observability (and benchmarking).

```bash
run_c13_results.ipynb
```


## 6) Requirements
Python ≥ 3.10
```bash
conda create -n invfba python=3.10 -y
conda activate invfba
pip install numpy pandas torch matplotlib joblib cobra cvxpy cvxpylayers gpytorch 