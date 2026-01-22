import argparse
import math
import os
import logging
import cvxpy as cp
import numpy as np
import torch
import cobra
from scipy.optimize import minimize
from cobra import Model, Reaction
from cobra.io import load_json_model
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set, Any, Union
import numpy as np
import cvxpy as cp
import gurobipy as gp
from cobra.util.solver import set_objective

# Force COBRApy/optlang to use a non-gurobi backend to avoid API mismatch
cobra.Configuration().solver = "glpk"


# TODO: add experimental error to measured
# todo: remove producti from obj find


def thesis_solve_objfind(
    S: np.ndarray,
    v_exp: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    measured_flux_idx: List[int],
    objective_flux_idx: List[int],
    S_v: float,
    rho: float = 1e-3,
    solver: str = "SCS",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve ObjFind by minimizing squared error to experimental fluxes,
    subject to optimality constraints.

    Parameters:
        S: Stoichiometric matrix (m x n)
        v_exp: Experimental flux vector (n,)
        v_min: Lower bounds for fluxes (n,)
        v_max: Upper bounds for fluxes (n,)
        measured_flux_indices: Indices j where v_exp[j] is measured
        objective_flux_indices: Indices j used in objective c_j v_j
        epsilon: Tolerance for relaxed complementarity conditions
        solver: CVXPY solver name (e.g., "SCS", "ECOS")

    Returns:
        v_opt: Optimal flux vector (n,)
        c_opt: Inferred objective vector (n,)
        loss: Value of squared error objective
    """
    m, n = S.shape

    # ── variables ────────────────────────────────────────────────
    v = cp.Variable(n)
    c = cp.Variable(n)
    lam = cp.Variable(m)
    mu_lb = cp.Variable(n, nonneg=True)
    mu_ub = cp.Variable(n, nonneg=True)
    s_lb = cp.Variable(n, nonneg=True)  # complementarity surrogates
    s_ub = cp.Variable(n, nonneg=True)

    obj_mask = np.isin(np.arange(n), objective_flux_idx)

    # ── constraints ──────────────────────────────────────────────
    constraints = [
        # primal feasibility
        S @ v == S_v,
        v >= v_min,
        v <= v_max,
        # stationarity
        c - S.T @ lam - mu_lb + mu_ub == 0,
        # convex complementarity surrogate
        mu_lb <= s_lb,
        v - v_min <= s_lb,
        mu_ub <= s_ub,
        v_max - v <= s_ub,
        # identifiability of c
        c[~obj_mask] == 0,
        c[obj_mask] >= 0,
        cp.sum(c[obj_mask]) == 1,
    ]

    # ── objective: data-fit + complementarity penalty ────────────
    data_fit = cp.sum_squares(v[measured_flux_idx] - v_exp[measured_flux_idx])
    objective = cp.Minimize(data_fit + rho * (cp.sum(s_lb) + cp.sum(s_ub)))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=True)

    print(f"Status: {prob.status}")
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("⚠️ Problem not solved optimally.")
        print("Objective value:", prob.value)
        print("Infeasibility certificate:", prob.status)
        if v.value is not None and c.value is not None:
            print("⚠️ Infeasible, but retrieving current best estimate.")
            return v.value, c.value, prob.value
        else:
            print("🚫 Infeasible and no solution returned.")
            return np.full(n, np.nan), np.full(n, np.nan), np.inf

    return v.value, c.value, prob.value


def true_solve_objfind(
    S: np.ndarray,
    v_exp: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    measured_flux_idx: List[int],
    objective_flux_idx: List[int],
    S_v: float,  # unused; kept for drop-in compatibility
    eps_ss: float = 1e-3,
    M_primal_bnd: float = 2_000.0,
    M_primal_ss: float = 10.0,
    M_dual: float = 1e6,
    solver: str = "GUROBI",
    return_diagnostics: bool = True,
    slack_tol: float = 1e-9,
) -> Union[
    Tuple[np.ndarray, np.ndarray, float],
    Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]],
]:
    """
    MIQP KKT benchmark with fixed steady-state relaxation:
        min_v c^T v
        s.t. -eps_ss <= S v <= eps_ss, v_min <= v <= v_max
    Single-level (KKT) enforced via binaries; top-level fits measured fluxes.
    """
    m: int
    n: int
    m, n = S.shape

    # --- basic input checks ---
    if eps_ss <= 0.0 or not np.isfinite(eps_ss):
        raise ValueError("eps_ss must be positive and finite.")
    if not (np.isfinite(S).all() and np.isfinite(v_exp).all() and np.isfinite(v_min).all() and np.isfinite(v_max).all()):
        out_v = np.full(n, np.nan)
        out_c = np.full(n, np.nan)
        diag: Dict[str, Any] = {"status": "input_nonfinite"}
        return (out_v, out_c, float("inf"), diag) if return_diagnostics else (out_v, out_c, float("inf"))

    # --- scaling SS constraints to improve numerics: (S/eps) v <= 1 ---
    A: np.ndarray = S / float(eps_ss)

    v: cp.Variable = cp.Variable(n)
    c: cp.Variable = cp.Variable(n)

    # duals
    y_pos: cp.Variable = cp.Variable(m, nonneg=True)  # A v <= 1
    y_neg: cp.Variable = cp.Variable(m, nonneg=True)  # -A v <= 1
    mu_lb: cp.Variable = cp.Variable(n, nonneg=True)  # v_min - v <= 0
    mu_ub: cp.Variable = cp.Variable(n, nonneg=True)  # v - v_max <= 0

    # binaries (complementarity)
    z_pos: cp.Variable = cp.Variable(m, boolean=True)
    z_neg: cp.Variable = cp.Variable(m, boolean=True)
    z_lb: cp.Variable = cp.Variable(n, boolean=True)
    z_ub: cp.Variable = cp.Variable(n, boolean=True)

    obj_mask: np.ndarray = np.isin(np.arange(n), objective_flux_idx)
    if not np.any(obj_mask):
        raise ValueError("objective_flux_idx selects no reactions.")

    constraints: List[cp.Constraint] = []

    # primal feasibility
    constraints += [
        A @ v <= 1.0,
        -A @ v <= 1.0,
        v >= v_min,
        v <= v_max,
    ]

    # stationarity (scaled: uses A^T)
    constraints += [
        c + A.T @ y_pos - A.T @ y_neg - mu_lb + mu_ub == 0
    ]

    # c identifiability
    constraints += [
        c[~obj_mask] == 0,
        c[obj_mask] >= 0,
        cp.sum(c[obj_mask]) == 1,
    ]

    # complementarity big-M (scaled SS slacks are O(1))
    ss_slack_pos: cp.Expression = 1.0 - (A @ v)     # >= 0
    ss_slack_neg: cp.Expression = 1.0 + (A @ v)     # >= 0
    lb_slack: cp.Expression = v - v_min             # >= 0
    ub_slack: cp.Expression = v_max - v             # >= 0

    constraints += [
        y_pos <= float(M_dual) * z_pos,
        ss_slack_pos <= float(M_primal_ss) * (1 - z_pos),

        y_neg <= float(M_dual) * z_neg,
        ss_slack_neg <= float(M_primal_ss) * (1 - z_neg),

        mu_lb <= float(M_dual) * z_lb,
        lb_slack <= float(M_primal_bnd) * (1 - z_lb),

        mu_ub <= float(M_dual) * z_ub,
        ub_slack <= float(M_primal_bnd) * (1 - z_ub),
    ]

    # top-level fit
    data_fit: cp.Expression = cp.sum_squares(v[measured_flux_idx] - v_exp[measured_flux_idx])
    prob: cp.Problem = cp.Problem(cp.Minimize(data_fit), constraints)

    # GUROBI parameters (reasonable defaults for MIQP)
    gurobi_opts: Dict[str, Any] = {
        "MIPGap": 1e-6,
        "FeasibilityTol": 1e-8,
        "OptimalityTol": 1e-8,
        "NumericFocus": 3,
        "Presolve": 2,
        "Threads": 0,         # let GUROBI choose
        "TimeLimit": 300.0,   # adjust as needed
        "OutputFlag": 1,
    }

    prob.solve(solver=solver,  reoptimize=True, verbose=True, **gurobi_opts)

    if prob.status not in ("optimal", "optimal_inaccurate") or v.value is None or c.value is None:
        out_v = np.full(n, np.nan)
        out_c = np.full(n, np.nan)
        diag = {"status": str(prob.status)}
        return (out_v, out_c, float("inf"), diag) if return_diagnostics else (out_v, out_c, float("inf"))

    v_val: np.ndarray = np.asarray(v.value).reshape(-1)
    c_val: np.ndarray = np.asarray(c.value).reshape(-1)
    loss_val: float = float(prob.value)

    if not return_diagnostics:
        return v_val, c_val, loss_val

    Sv: np.ndarray = (S @ v_val).reshape(-1)

    diagnostics: Dict[str, Any] = {
        "status": str(prob.status),
        "eps_ss": float(eps_ss),
        "max_abs_Sv": float(np.max(np.abs(Sv))),
        "n_active_lb": int(np.sum((v_val - v_min) <= slack_tol)),
        "n_active_ub": int(np.sum((v_max - v_val) <= slack_tol)),
    }
    return v_val, c_val, loss_val, diagnostics


def solve_objfind(
    S: np.ndarray,
    v_exp: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    measured_flux_idx: List[int],
    objective_flux_idx: List[int],
    eps_ss: float,
    rho_kkt: float = 1e-3,
    solver: str = "SCS",
    return_diagnostics: bool = True,
    slack_tol: float = 1e-9,
) -> Union[
    Tuple[np.ndarray, np.ndarray, float],
    Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]],
]:
    """
    Convex single-level *relaxation* of ObjFind (not MIQP-exact):

      min_v,c,duals  ||v_M - vexp_M||^2  +  rho_kkt * KKT_tightness(v,duals)
      s.t.           -eps_ss <= S v <= eps_ss
                     v_min <= v <= v_max
                     stationarity for L = c^T v + y_pos^T(Sv-eps) + y_neg^T(-Sv-eps) + mu_lb^T(v_min-v) + mu_ub^T(v-v_max)
                     c supported only on objective_flux_idx and on simplex

    Here, KKT_tightness is an L1 proxy that penalizes (i) dual magnitudes and
    (ii) primal slacks for bounds and steady-state inequalities. This reduces
    "too perfect" fits by discouraging solutions that rely on loose KKT behavior.

    Args:
        S: Stoichiometric matrix (m, n)
        v_exp: Experimental flux vector (n,)
        v_min: Lower bounds (n,)
        v_max: Upper bounds (n,)
        measured_flux_idx: indices used in the data-fit term
        objective_flux_idx: indices allowed to have nonzero coefficients in c
        eps_ss: steady-state tolerance used as -eps_ss <= S v <= eps_ss
        rho_kkt: weight on KKT tightness penalty
        solver: CVXPY solver name (e.g., "SCS", "ECOS")
        return_diagnostics: whether to return a diagnostics dict
        slack_tol: threshold for counting "active" constraints

    Returns:
        (v_opt, c_opt, loss) or (v_opt, c_opt, loss, diagnostics)
    """
    m: int
    n: int
    m, n = S.shape

    # --- variables ---
    v: cp.Variable = cp.Variable(n)
    c: cp.Variable = cp.Variable(n)

    # duals for inequalities (all nonnegative)
    y_pos: cp.Variable = cp.Variable(m, nonneg=True)  # Sv - eps_ss <= 0
    y_neg: cp.Variable = cp.Variable(m, nonneg=True)  # -Sv - eps_ss <= 0
    mu_lb: cp.Variable = cp.Variable(n, nonneg=True)  # v_min - v <= 0
    mu_ub: cp.Variable = cp.Variable(n, nonneg=True)  # v - v_max <= 0

    # --- masks / identifiability ---
    obj_mask: np.ndarray = np.isin(np.arange(n), objective_flux_idx)
    if not np.any(obj_mask):
        raise ValueError("objective_flux_idx selects no reactions.")

    # --- primal feasibility (inequality form) ---
    g_pos: cp.Expression = (S @ v) - float(eps_ss)      # <= 0
    g_neg: cp.Expression = (-S @ v) - float(eps_ss)     # <= 0
    g_lb: cp.Expression = v_min - v                     # <= 0  (v >= v_min)
    g_ub: cp.Expression = v - v_max                     # <= 0  (v <= v_max)

    constraints: List[cp.Constraint] = [
        g_pos <= 0,
        g_neg <= 0,
        g_lb <= 0,
        g_ub <= 0,
    ]

    # --- stationarity ---
    # L = c^T v + y_pos^T(Sv-eps) + y_neg^T(-Sv-eps) + mu_lb^T(v_min-v) + mu_ub^T(v-v_max)
    # dL/dv = c + S^T y_pos - S^T y_neg - mu_lb + mu_ub = 0
    constraints += [
        c + S.T @ y_pos - S.T @ y_neg - mu_lb + mu_ub == 0
    ]

    # --- c simplex on selected support ---
    constraints += [
        c[~obj_mask] == 0,
        c[obj_mask] >= 0,
        cp.sum(c[obj_mask]) == 1,
    ]

    # --- data fit ---
    data_fit: cp.Expression = cp.sum_squares(v[measured_flux_idx] - v_exp[measured_flux_idx])

    # --- KKT tightness proxy (L1) ---
    # primal slacks (all >= 0 by construction)
    ss_slack_pos: cp.Expression = float(eps_ss) - (S @ v)   # >= 0 when g_pos<=0
    ss_slack_neg: cp.Expression = float(eps_ss) + (S @ v)   # >= 0 when g_neg<=0
    lb_slack: cp.Expression = v - v_min                     # >= 0
    ub_slack: cp.Expression = v_max - v                     # >= 0

    kkt_pen: cp.Expression = (
        (cp.sum(y_pos) + cp.sum(y_neg)) / float(m)
        + (cp.sum(mu_lb) + cp.sum(mu_ub)) / float(n)
        + (cp.sum(ss_slack_pos) + cp.sum(ss_slack_neg)) / float(m)
        + (cp.sum(lb_slack) + cp.sum(ub_slack)) / float(n)
    )

    objective: cp.Minimize = cp.Minimize(data_fit + float(rho_kkt) * kkt_pen)
    prob: cp.Problem = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=True)

    if prob.status not in ("optimal", "optimal_inaccurate") or v.value is None or c.value is None:
        out_v: np.ndarray = np.full(n, np.nan)
        out_c: np.ndarray = np.full(n, np.nan)
        loss_val: float = float("inf")
        if not return_diagnostics:
            return out_v, out_c, loss_val
        fail_diag: Dict[str, Any] = {
            "status": str(prob.status),
            "eps_ss": float(eps_ss),
            "rho_kkt": float(rho_kkt),
            "max_abs_Sv": float("nan"),
            "kkt_pen": float("nan"),
            "n_active_lb": 0,
            "n_active_ub": 0,
        }
        return out_v, out_c, loss_val, fail_diag

    v_val: np.ndarray = np.asarray(v.value).reshape(-1)
    c_val: np.ndarray = np.asarray(c.value).reshape(-1)
    loss_val = float(prob.value)

    if not return_diagnostics:
        return v_val, c_val, loss_val

    Sv_val: np.ndarray = (S @ v_val).reshape(-1)
    max_abs_Sv: float = float(np.max(np.abs(Sv_val))) if Sv_val.size > 0 else 0.0

    # activity counts (near-active bounds)
    n_active_lb: int = int(np.sum((v_val - v_min) <= float(slack_tol)))
    n_active_ub: int = int(np.sum((v_max - v_val) <= float(slack_tol)))

    # evaluate penalty value (best-effort)
    try:
        kkt_pen_val: float = float(kkt_pen.value)  # type: ignore[assignment]
    except Exception:
        kkt_pen_val = float("nan")

    diagnostics: Dict[str, Any] = {
        "status": str(prob.status),
        "eps_ss": float(eps_ss),
        "rho_kkt": float(rho_kkt),
        "max_abs_Sv": max_abs_Sv,
        "kkt_pen": kkt_pen_val,
        "n_active_lb": n_active_lb,
        "n_active_ub": n_active_ub,
    }
    return v_val, c_val, loss_val, diagnostics


def project_simplex(c):
    """Project vector c to the probability simplex (non-negative, sum to 1)."""
    c = np.maximum(c, 0)
    if np.sum(c) == 0:
        return np.ones_like(c) / len(c)
    return c / np.sum(c)


def build_selector_matrix(measured_indices, total_fluxes):
    """
    Create a selector matrix that maps a small objective vector (only measured fluxes)
    into the full flux space.
    """
    selector = np.zeros((total_fluxes, len(measured_indices)))
    for i, idx in enumerate(measured_indices):
        selector[idx, i] = 1
    return selector


def objfind_loss_with_selector(c_small, selector, S, v_min, v_max, v_exp, measured_indices):
    c_small_proj = project_simplex(c_small)
    c_full = selector @ c_small_proj
    v_pred = inner_fba(c_full, S, v_min, v_max)
    if v_pred is None:
        return 1e6  # Penalize solver failure
    v_pred_measured = v_pred[measured_indices]
    v_exp_measured = v_exp[measured_indices]
    return np.sum((v_pred_measured - v_exp_measured)**2)



def inner_fba(c_full: np.ndarray, S: np.ndarray, v_min: np.ndarray, v_max: np.ndarray, S_v: float = 0.0) -> np.ndarray:
    v = cp.Variable(len(c_full))
    constraints = [S @ v == S_v, v >= v_min, v <= v_max]
    prob = cp.Problem(cp.Maximize(c_full @ v), constraints)
    prob.solve(solver=cp.SCS)
    if prob.status != cp.OPTIMAL:
        print(f"FBA failed with status: {prob.status}")
        return None
    return v.value if v.value is not None else None



def load_experimental_fluxes(model, flux_to_index, flux_scale_map, fluxes_cols, sheet, selected_index=1):
    flux_array = torch.zeros(len(model.reactions))
    experimental_fluxes_row = sheet.loc[selected_index, fluxes_cols]

    for flux_name in fluxes_cols:
        value = experimental_fluxes_row[flux_name]
        if flux_name in flux_scale_map:
            value *= flux_scale_map[flux_name]
        reaction_index = flux_to_index[flux_name]
        flux_array[reaction_index] = value

    return flux_array

def tighten_flux_bounds(model, experimental_fluxes_tensor, flux_to_index, flux_scale_map, fluxes_cols, objective_fluxes):
    v_min = torch.full((len(model.reactions),), -1000.0)
    v_max = torch.full((len(model.reactions),), 1000.0)

    for flux_name in fluxes_cols:
        value = experimental_fluxes_tensor[flux_to_index[flux_name]]
        if flux_name in flux_scale_map:
            value *= flux_scale_map[flux_name]
        reaction_index = flux_to_index[flux_name]

        if flux_name in objective_fluxes:
            #eps = max(1e-1, 0.15 * abs(value))
            eps = 0.15 * abs(value)
            v_min[reaction_index] = value - eps
            v_max[reaction_index] = value + eps
            print(f"🔒 Tight bounds for {flux_name}: {v_min[reaction_index]:.4g} to {v_max[reaction_index]:.4g}")
        else:
            print(f"🔓 Wide bounds for {flux_name}: {v_min[reaction_index]:.4g} to {v_max[reaction_index]:.4g}")

    return v_min, v_max

def initialize_S_and_vs(model) -> Tuple[np.ndarray, List[float], List[float], torch.Tensor, torch.Tensor, torch.Tensor]:
    S = np.zeros((len(model.metabolites), len(model.reactions)))

    for i, metabolite in enumerate(model.metabolites):
        for j, reaction in enumerate(model.reactions):
            if metabolite in reaction.metabolites:
                S[i, j] = reaction.metabolites[metabolite]

    v_min = []
    v_max = []

    for rxn in model.reactions:
        v_min.append(-1000.0 if rxn.lower_bound < 0 else 0.0)
        v_max.append(1000.0)

    S_torch = torch.tensor(S, dtype=torch.float32)
    v_min_torch = torch.tensor(v_min, dtype=torch.float32)
    v_max_torch = torch.tensor(v_max, dtype=torch.float32)

    return S, v_min, v_max, S_torch, v_min_torch, v_max_torch

# Mapping from experiment ID and day to v_exp
def get_v_exp(excel_path: str, experiment_id: str, day: int) -> np.ndarray:
    df = pd.read_excel(excel_path, sheet_name=experiment_id)
    v_exp_all = df.set_index("Day").loc[day].values
    return v_exp_all.astype(float)


def solve_forward_fba(
    S: np.ndarray,
    c: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    S_v: float,
    rho_ss: float = 1e2,
    rho_bnd: float = 1e2,
    solver: str = "ECOS",
) -> Tuple[np.ndarray, str]:
    """
    Forward solve consistent with slackened feasibility:
      min c^T v + rho_ss*||s_S||_1 + rho_bnd*(||s_lb||_1+||s_ub||_1)
      s.t. Sv <= S_v + s_S, -Sv <= S_v + s_S
           v >= v_min - s_lb, v <= v_max + s_ub
           slacks >= 0
    """
    m: int
    n: int
    m, n = S.shape

    v: cp.Variable = cp.Variable(n)
    s_S: cp.Variable = cp.Variable(m, nonneg=True)
    s_lb: cp.Variable = cp.Variable(n, nonneg=True)
    s_ub: cp.Variable = cp.Variable(n, nonneg=True)

    c_clean: np.ndarray = np.maximum(np.asarray(c).reshape(-1), 0.0)
    s: float = float(c_clean.sum())
    if s > 0.0:
        c_clean = c_clean / s

    constraints = [
        S @ v <= S_v + s_S,
        -S @ v <= S_v + s_S,
        v >= v_min - s_lb,
        v <= v_max + s_ub,
    ]

    objective: cp.Minimize = cp.Minimize(
        c_clean @ v
        + rho_ss * (cp.sum(s_S) / float(m))
        + rho_bnd * ((cp.sum(s_lb) + cp.sum(s_ub)) / float(n))
    )

    prob: cp.Problem = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=True)

    if prob.status not in ("optimal", "optimal_inaccurate") or v.value is None:
        return np.full(n, np.nan), str(prob.status)

    return np.asarray(v.value).reshape(-1), str(prob.status)


def hard_solve_forward_fba(
    S: np.ndarray,
    c: np.ndarray,
    v_min: Optional[np.ndarray],
    v_max: Optional[np.ndarray],
    S_v: float,
    solver: str = "OSQP",
) -> Tuple[Optional[np.ndarray], str]:
    """
    Forward solve consistent with solve_objfind:
      minimize c^T v
      s.t. -S_v <= S v <= S_v, v_min <= v <= v_max
    """
    m, n = S.shape
    v = cp.Variable(n)

    constraints = [
        S @ v <= S_v,
        -S @ v <= S_v,
    ]

    if v_min is not None:
        constraints += [v >= v_min]
    else:
        constraints += [v >= 0.0]

    if v_max is not None:
        constraints += [v <= v_max]

    # sanitize tiny negatives from SCS etc. (optional but robust)
    c_clean = np.maximum(c, 0.0)
    s = float(c_clean.sum())
    if s > 0:
        c_clean = c_clean / s

    prob = cp.Problem(cp.Minimize(c_clean @ v), constraints)
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"] or v.value is None:
        return None, str(prob.status)

    return np.asarray(v.value).reshape(-1), str(prob.status)


def load_model(model_path):
    """Load the metabolic model from a JSON file."""
    try:
        model = cobra.io.load_json_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def constrain_model(model, lb, ub, median):
    for rxn in model.reactions:
        rxn_id = rxn.id

        # Only adjust if rxn_id is present in bounds
        if rxn_id in lb.index and rxn_id in ub.index:
            new_lb = lb[rxn_id]
            new_ub = ub[rxn_id]

            print(f"🔧 Adjusted bounds for {rxn_id}")
            print(f"  - LB (final): {new_lb}")
            print(f"  - UB (final): {new_ub}")

            rxn.bounds = (new_lb, new_ub)
            # rxn.lower_bound = new_lb
            # rxn.upper_bound = new_ub

    # Optional: access median values if needed
    #median_fluxes = np.array([median[rxn.id] for rxn in model.reactions if rxn.id in median.index])

    return model



def extract_bounds(
    flux_df: pd.DataFrame,
    model: Model,
    condition: str,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      lb_row: float Series aligned to model.reactions (index = rxn.id)
      ub_row: float Series aligned to model.reactions (index = rxn.id)
      median_row: float Series aligned to model.reactions, with NaNs filled by 0.0 (numeric v_exp)
      median_measured_mask: bool Series aligned to model.reactions, True where median was present in file
    """
    df: pd.DataFrame = flux_df.iloc[1:].copy()
    df.set_index("Reaction", inplace=True)
    df.index = df.index.astype(str).str.strip()

    lb_key: str = f"{condition}_LB"
    ub_key: str = f"{condition}_UB"
    med_key: str = f"{condition}_MEDIAN"

    lb_raw = df.loc[lb_key] if lb_key in df.index else None
    ub_raw = df.loc[ub_key] if ub_key in df.index else None
    med_raw = df.loc[med_key] if med_key in df.index else None

    cobra_rxns: list[str] = [rxn.id for rxn in model.reactions]

    # --- LB/UB: keep  defaults ---
    if lb_raw is None or ub_raw is None:
        lb_row = pd.Series(
            {rxn.id: (-1000.0 if rxn.reversibility else 0.0) for rxn in model.reactions},
            dtype=float,
        )
        ub_row = pd.Series({rxn.id: 1000.0 for rxn in model.reactions}, dtype=float)
    else:
        lb_row = lb_raw.astype(float).fillna(-1000.0)
        ub_row = ub_raw.astype(float).fillna(1000.0)

    lb_row = lb_row.reindex(cobra_rxns, fill_value=-1000.0).astype(float)
    ub_row = ub_row.reindex(cobra_rxns, fill_value=1000.0).astype(float)

    # --- MEDIAN: preserve NaNs to define "measured" ---
    if med_raw is None:
        med_aligned = pd.Series(index=cobra_rxns, data=pd.NA, dtype="float64")
    else:
        med_aligned = med_raw.astype(float).reindex(cobra_rxns)  # keep NaNs

    median_measured_mask: pd.Series = med_aligned.notna().astype(bool)
    median_row: pd.Series = med_aligned.fillna(0.0).astype(float)  # numeric v_exp

    return lb_row, ub_row, median_row, median_measured_mask


def initialize_S_and_vs(
    model,
    lb_row: pd.Series,
    ub_row: pd.Series
) -> Tuple[np.ndarray, List[float], List[float], torch.Tensor, torch.Tensor, torch.Tensor]:
    # === Build stoichiometric matrix ===
    S = np.zeros((len(model.metabolites), len(model.reactions)))
    for i, metabolite in enumerate(model.metabolites):
        for j, reaction in enumerate(model.reactions):
            if metabolite in reaction.metabolites:
                S[i, j] = reaction.metabolites[metabolite]

    # === Use extracted bounds ===
    cobra_rxns = [rxn.id for rxn in model.reactions]
    v_min = [lb_row[rxn_id] for rxn_id in cobra_rxns]
    v_max = [ub_row[rxn_id] for rxn_id in cobra_rxns]

    # === Convert to torch tensors ===
    S_torch = torch.tensor(S, dtype=torch.float32)
    v_min_torch = torch.tensor(v_min, dtype=torch.float32)
    v_max_torch = torch.tensor(v_max, dtype=torch.float32)

    return S, v_min, v_max, S_torch, v_min_torch, v_max_torch



def enforce_non_producer_zero_flux(
    model: Model,
    lb: pd.Series,
    ub: pd.Series,
    condition: str,
    non_producer_conditions: Set[str],
    product_reactions: List[str],
) -> Tuple[Model, pd.Series, pd.Series]:
    """
    For non-producer conditions, sets product fluxes to 0 in both model and bounds.
    """
    if condition not in non_producer_conditions:
        return model, lb, ub

    for rxn_id in product_reactions:
        if rxn_id not in lb.index:
            continue
        # update model
        rxn = model.reactions.get_by_id(rxn_id)
        rxn.lower_bound = 0.0
        rxn.upper_bound = 0.0
        # update vectors
        lb.at[rxn_id] = 0.0
        ub.at[rxn_id] = 0.0

    return model, lb, ub


def check_feasible(
    S: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    S_v: float,
    solver: str = "ECOS",
) -> str:
    n = S.shape[1]
    v = cp.Variable(n)
    constraints = [S @ v <= S_v, -S @ v <= S_v, v >= v_min, v <= v_max]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=solver, verbose=True)
    return str(prob.status)



if __name__ == '__main__':
    # === Load and constrain model ===

    parser = argparse.ArgumentParser(description="Starting inverse FBA.")
    parser.add_argument("--fluxes_mode", type=str, default="standard", help="Fluxes in objective function (default: standard).")
    args = parser.parse_args()

    conditions =  ["AWZ11expo", "AWZ11stat", "early", "stat"]

    S_v = 0.001
    #S_v = 0

    records = []

    for condition in conditions:
        model = load_model('inputs/CHOmpact_cobrapy_original.json')
        file_path = 'inputs/C13_data.xlsx'
        c13 = pd.read_excel(file_path, sheet_name="c13")

        # === Output directory ===
        output_csv = f"NPJ/cho/benchmark/{args.fluxes_mode}/"
        os.makedirs(output_csv, exist_ok=True)

        num_reactions = len(model.reactions)

        NON_PRODUCER_CONDITIONS: Set[str] = {"AWZ11expo", "AWZ11stat"}
        PRODUCT_REACTIONS: List[str] = ["F143"]

        STANDARD_OBJECTIVES: Dict[str, List[str]] = {
            "AWZ11expo": ["F142", "F105", "F107", "F112", "F114"],
            "AWZ11stat": ["F142", "F105", "F107", "F112", "F114"],
            "early": ["F142", "F143", "F105", "F107", "F112", "F114"],
            "stat": ["F142", "F143", "F105", "F107", "F112", "F114"],
        }

        EXTRACELLULAR_OBJ: Dict[str, List[str]] = {
            "AWZ11expo": ['F110', 'F111', 'F112', 'F113', 'F142', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118', 'F107', 'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127', 'F128'],
            "AWZ11stat": ['F110', 'F111', 'F112', 'F113', 'F142', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118', 'F107', 'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127','F128'],
            "early": ['F110', 'F143', 'F111', 'F112', 'F113', 'F142', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118', 'F107', 'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127','F128'],
            "stat": ['F110', 'F143', 'F111', 'F112', 'F113', 'F142', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118', 'F107', 'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127', 'F128'],
        }

        MEASURED_OBJ: Dict[str, List[str]] = {
            "AWZ11expo": ['F2', 'F3', 'F4', 'F5', 'F6', 'F1', 'F25', 'F27', 'F7', 'F8', 'F13', 'F14', 'F15', 'F9', 'F11', 'F10', 'F17', 'F16', 'F32', 'F52', 'F51', 'F34', 'F48', 'F54', 'F38', 'F40',
                          'F40', 'F23', 'F26', 'F44', 'F45', 'F49', 'F41', 'F39', 'F31', 'F57', 'F55', 'F110', 'F111', 'F112', 'F113', 'F142', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118', 'F107',
                          'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127', 'F128'],
            "AWZ11stat": ['F2', 'F3', 'F4', 'F5', 'F6', 'F1', 'F25', 'F27', 'F7', 'F8', 'F13', 'F14', 'F15', 'F9', 'F11', 'F10', 'F17', 'F16', 'F32', 'F52', 'F51', 'F34', 'F48', 'F54', 'F38', 'F40',
                          'F40', 'F23', 'F26', 'F44', 'F45', 'F49', 'F41', 'F39', 'F31', 'F57', 'F55', 'F110', 'F111', 'F112', 'F113', 'F142', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118', 'F107',
                          'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127', 'F128'],
            "early": ['F2', 'F3', 'F4', 'F5', 'F6', 'F1', 'F25', 'F27', 'F7', 'F8', 'F13', 'F14', 'F15', 'F9', 'F11', 'F10', 'F17', 'F16', 'F32', 'F52', 'F51', 'F34', 'F48', 'F54', 'F38', 'F40',
                      'F40', 'F23', 'F26', 'F44', 'F45', 'F49', 'F41', 'F39', 'F31', 'F57', 'F55', 'F110', 'F111', 'F112', 'F113', 'F142', 'F143', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118',
                      'F107', 'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127', 'F128'],
            "stat": ['F2', 'F3', 'F4', 'F5', 'F6', 'F1', 'F25', 'F27', 'F7', 'F8', 'F13', 'F14', 'F15', 'F9', 'F11', 'F10', 'F17', 'F16', 'F32', 'F52', 'F51', 'F34', 'F48', 'F54', 'F38', 'F40', 'F40',
                     'F23', 'F26', 'F44', 'F45', 'F49', 'F41', 'F39', 'F31', 'F57', 'F55', 'F110', 'F111', 'F112', 'F113', 'F142', 'F143', 'F105', 'F114', 'F115', 'F116', 'F117', 'F118', 'F107',
                     'F119', 'F120', 'F121', 'F122', 'F123', 'F124', 'F125', 'F126', 'F127', 'F128'],
        }

        if args.fluxes_mode == "all":
            objective_fluxes: List[str] = [f"F{i}" for i in range(1, 145)]
        elif args.fluxes_mode == "standard":
            objective_fluxes = STANDARD_OBJECTIVES[condition]
        elif args.fluxes_mode == "extra":
            objective_fluxes = EXTRACELLULAR_OBJ[condition]
        elif args.fluxes_mode == "measured":
            objective_fluxes = MEASURED_OBJ[condition]
        else:
            raise ValueError(f"Unknown fluxes mode: {args.fluxes_mode}")

        flux_to_index = {rxn.id: i for i, rxn in enumerate(model.reactions)}
        objective_fluxes_indices = [flux_to_index[name] for name in objective_fluxes]

        lb_row, ub_row, median_row, median_measured_mask = extract_bounds(c13, model, condition)

        model, lb_row, ub_row = enforce_non_producer_zero_flux(model, lb_row, ub_row, condition, NON_PRODUCER_CONDITIONS, PRODUCT_REACTIONS)
        constrained_model = constrain_model(model, lb_row, ub_row, median_row)

        # === Stoichiometric setup ===
        S, vmin, vmax, S_torch, v_min_torch, v_max_torch = initialize_S_and_vs(constrained_model, lb_row, ub_row)
        objective_flux_indices = [flux_to_index[f] for f in objective_fluxes]
        median_tensor = torch.tensor(median_row.values, dtype=torch.float32)

        measured_flux_indices: list[int] = []
        for rxn_id, is_measured in median_measured_mask.items():
            if not bool(is_measured):
                continue
            j: int = flux_to_index[rxn_id]
            if math.isfinite(float(median_tensor[j].item())):
                measured_flux_indices.append(j)

        measured_fluxes: list[str] = [str(rxn_id) for rxn_id, is_measured in median_measured_mask.items() if bool(is_measured)]

        v_opt, c_opt, loss, diagnostics = solve_objfind(
            S=S,
            v_exp=median_tensor.numpy(),
            v_min=v_min_torch.numpy(),
            v_max=v_max_torch.numpy(),
            measured_flux_idx=measured_flux_indices,
            objective_flux_idx=[flux_to_index[f] for f in objective_fluxes],
            eps_ss=S_v
        )

        feas_status = check_feasible(S, v_min_torch.numpy(), v_max_torch.numpy(), S_v)
        print("condition:", condition, "feasible?", feas_status)

        v_fba, fwd_status = solve_forward_fba(S, c_opt, v_min_torch, v_max_torch, S_v)

        objective_fluxes = list(dict.fromkeys(objective_fluxes))  # dedup
        objective_idx: list[int] = [flux_to_index[f] for f in objective_fluxes]

        measured_fluxes_all: list[str] = [
            str(rxn_id) for rxn_id, ok in median_measured_mask.items() if bool(ok)
        ]
        measured_idx_all: list[int] = [flux_to_index[rid] for rid in measured_fluxes_all]

        # only compare where (a)  allow c to exist and (b) have data
        # fit_idx: list[int] = sorted(set(objective_idx).intersection(measured_idx_all))
        # fit_fluxes: list[str] = [model.reactions[i].id for i in fit_idx]  # stable ids

        fit_idx_raw: list[int] = sorted(set(objective_idx).intersection(measured_idx_all))
        fit_idx: list[int] = [
            j for j in fit_idx_raw
            if math.isfinite(float(median_tensor[j].item()))
        ]
        fit_fluxes: list[str] = [model.reactions[j].id for j in fit_idx]

        record: dict[str, Any] = {
            "condition": condition,
            "fluxes_mode": args.fluxes_mode,

            # optimization outcome
            "status": str(diagnostics.get("status", "missing")),
            "is_feasible": diagnostics.get("status", "") in ("optimal", "optimal_inaccurate"),
            "loss": float(loss) if np.isfinite(loss) else float("inf"),

            # steady-state handling (fairness parameter)
            "eps_ss": float(diagnostics.get("eps_ss", float("nan"))),

            # feasibility / physics diagnostics
            "max_abs_Sv": float(diagnostics.get("max_abs_Sv", float("nan"))),
            "n_active_lb": int(diagnostics.get("n_active_lb", 0)),
            "n_active_ub": int(diagnostics.get("n_active_ub", 0)),

            # relaxation-specific (important to explain “too good” fits)
            "rho_kkt": float(diagnostics.get("rho_kkt", float("nan"))),
            "kkt_pen": float(diagnostics.get("kkt_pen", float("nan"))),
        }

        record.update({f"c_{rid}": float(c_opt[flux_to_index[rid]]) for rid in objective_fluxes})
        record.update({f"v_exp_{rid}": float(median_tensor[flux_to_index[rid]].item()) for rid in fit_fluxes})

        if v_fba is None:
            record.update({f"v_pred_{rid}": math.nan for rid in fit_fluxes})
        else:
            record.update({f"v_pred_{rid}": float(v_fba[flux_to_index[rid]]) for rid in fit_fluxes})

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(f"{output_csv}/prod_objfind_Sv{S_v}_.csv", index=False)
    print("Saved forward FBA results'")

