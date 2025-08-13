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
from typing import List, Tuple, Optional
import numpy as np
import cvxpy as cp



def solve_objfind(
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
        S @ v <= S_v,
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


# Optional: override this if your FBA solver changes constraint RHS
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



def solve_forward_fba(S, c, v_min, v_max, S_v):
    """
    Forward FBA using inferred objective vector `c` and optional lower/upper bounds.
    """
    M, N = S.shape
    v = cp.Variable(N)
    constraints = [S @ v <= S_v]

    if v_min is not None:
        constraints += [v >= v_min]
    else:
        constraints += [v >= 0]

    if v_max is not None:
        constraints += [v <= v_max]

    prob = cp.Problem(cp.Maximize(cp.sum(c * v)), constraints)
    prob.solve(solver=cp.SCS)

    return v.value, prob.status



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
    median_fluxes = np.array([median[rxn.id] for rxn in model.reactions if rxn.id in median.index])

    return model



def extract_bounds(flux_df, model, condition):
    # Skip enzyme row, keep flux data
    flux_df = flux_df.iloc[1:].copy()

    # Set the 'Reaction' column as the row index (to access early_LB etc.)
    flux_df.set_index("Reaction", inplace=True)
    flux_df.index = flux_df.index.astype(str).str.strip()  # Clean index

    lb_key = f"{condition}_LB"
    ub_key = f"{condition}_UB"
    median_key = f"{condition}_MEDIAN"

    lb_row = flux_df.loc[lb_key] if lb_key in flux_df.index else None
    ub_row = flux_df.loc[ub_key] if ub_key in flux_df.index else None
    median_row = flux_df.loc[median_key] if median_key in flux_df.index else None

    print("✅ Keys found:", lb_key in flux_df.index, ub_key in flux_df.index, median_key in flux_df.index)

    if lb_row is None or ub_row is None:
        print(f"⚠️ Missing LB/UB for {condition}. Using defaults!")
        lb_row = pd.Series({rxn.id: (-1000 if rxn.reversibility else 0) for rxn in model.reactions})
        ub_row = pd.Series({rxn.id: 1000 for rxn in model.reactions})
    else:
        lb_row = lb_row.fillna(-1000)
        ub_row = ub_row.fillna(1000)

    if median_row is None:
        print(f"⚠️ Missing MEDIAN for {condition}. Using 0's.")
        median_row = pd.Series({rxn.id: 0 for rxn in model.reactions})
    else:
        median_row = median_row.fillna(0)

    # Convert to float
    lb_row = lb_row.astype(float)
    ub_row = ub_row.astype(float)
    median_row = median_row.astype(float)

    # Align with model reactions
    cobra_rxns = [rxn.id for rxn in model.reactions]
    lb_row = lb_row.reindex(cobra_rxns, fill_value=-1000)
    ub_row = ub_row.reindex(cobra_rxns, fill_value=1000)
    median_row = median_row.reindex(cobra_rxns, fill_value=0)

    return lb_row, ub_row, median_row

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

    # === Use your extracted bounds ===
    cobra_rxns = [rxn.id for rxn in model.reactions]
    v_min = [lb_row[rxn_id] for rxn_id in cobra_rxns]
    v_max = [ub_row[rxn_id] for rxn_id in cobra_rxns]

    # === Convert to torch tensors ===
    S_torch = torch.tensor(S, dtype=torch.float32)
    v_min_torch = torch.tensor(v_min, dtype=torch.float32)
    v_max_torch = torch.tensor(v_max, dtype=torch.float32)

    return S, v_min, v_max, S_torch, v_min_torch, v_max_torch


if __name__ == '__main__':
    # === Load and constrain model ===
    conditions = ['early','stat'] #['AWZ11expo', 'AWZ11stat']# ['early','stat'] #   #

    S_v = 0.001
    #S_v = 0

    records = []


    for condition in conditions:
        model = load_model('inputs/CHOmpact_cobrapy_original.json')
        file_path = 'inputs/C13_data.xlsx'
        c13 = pd.read_excel(file_path, sheet_name="c13")

        # === Output directory ===
        output_csv = "Publication_Results/C13/ObjFind/"
        os.makedirs(output_csv, exist_ok=True)

        num_reactions = len(model.reactions)
        objective_fluxes = ['F142', 'F143', 'F105', 'F107', 'F112', 'F114'] # in prod'F143',
        flux_to_index = {rxn.id: i for i, rxn in enumerate(model.reactions)}
        objective_fluxes_indices = [flux_to_index[name] for name in objective_fluxes]

        lb_row, ub_row, median_row = extract_bounds(c13, model, condition)
        constrained_model = constrain_model(model, lb_row, ub_row, median_row)

        # === Stoichiometric setup ===
        S, vmin, vmax, S_torch, v_min_torch, v_max_torch = initialize_S_and_vs(
            constrained_model, lb_row, ub_row
        )
        objective_flux_indices = [flux_to_index[f] for f in objective_fluxes]
        median_tensor = torch.tensor(median_row.values, dtype=torch.float32)

        # After extracting median_row and constraining the model
        measured_fluxes = [rxn_id for rxn_id, val in median_row.items() if val != 0]
        measured_flux_indices = [flux_to_index[rxn_id] for rxn_id in measured_fluxes]


        v_opt, c_opt, loss = solve_objfind(
            S=S,
            v_exp=median_tensor.numpy(),
            v_min=v_min_torch.numpy(),
            v_max=v_max_torch.numpy(),
            measured_flux_idx=measured_flux_indices,
            objective_flux_idx=[flux_to_index[f] for f in objective_fluxes],
            S_v=S_v
        )

        v_fba, status = solve_forward_fba(S, c_opt, v_min_torch, v_max_torch, S_v)

        # After each solve:
        record = {
            "mode": condition,  # e.g., "fedbatch" or "continuous"
            "loss": loss,
            **{f"c_subset_{i}": c_opt[i] for i in range(len(c_opt))},
        }

        # Add measured fluxes
        if median_tensor is not None:
            v_measured = median_tensor.numpy().tolist()
            record.update({
                f"v_measured_{flux_to_index[name]}": median_tensor.numpy()[flux_to_index[name]]
                for name in measured_fluxes
            })

        # Add forward FBA-predicted fluxes
        if v_fba is not None:
            record.update({f"v_fba_{i}": v_fba[i] for i in sorted(range(len(v_fba)))})

        record["status"] = status

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(f"{output_csv}/objfind_Sv{S_v}.csv", index=False)
    print("✅ Saved forward FBA results'")

















