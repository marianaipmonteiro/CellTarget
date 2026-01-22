import os
from typing import Tuple

import cvxpy as cp
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer
from joblib import Parallel, delayed, parallel_backend

from bo import bayesian_optimization
from setup import FullObservabilityCase


def setup_optimization_with_basis_functions(case: FullObservabilityCase, mode: str = "norm") -> CvxpyLayer:
    v = cp.Variable(case.num_fluxes)
    C_param = cp.Parameter(case.num_fluxes)

    objective_fitness = cp.multiply(C_param, v)

    epsilon: float = 1e-6
    if mode == "norm":
        objective_expr = cp.norm(objective_fitness) + epsilon
    elif mode == "sum_squares":
        objective_expr = cp.sum_squares(objective_fitness)
    elif mode == "logsumexp":
        objective_expr = cp.log_sum_exp(objective_fitness)
    elif mode == "linear":
        objective_expr = cp.sum(objective_fitness)
    elif mode == "inverse_log":
        objective_expr = cp.log(cp.sum(objective_fitness) + epsilon)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Define the constraints
    constraints = [case.S @ v == 0, case.v_min <= v, v <= case.v_max]

    if mode in ["norm", "sum_squares", "logsumexp"]:
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    else: #inverse_log linear entropy
        problem = cp.Problem(cp.Maximize(objective_expr), constraints)

    return CvxpyLayer(problem, parameters=[C_param], variables=[v])


def setup_ground_truth(case: FullObservabilityCase):
    v = cp.Variable(case.num_fluxes)

    constraints = [case.S @ v == 0, v >= case.v_min, v <= case.v_max]
    objective = cp.Minimize(cp.norm(v))
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.SCS)
    optimal_fluxes = v.value

    print('Status:', problem.status)
    print('Optimal objective value:', problem.value)
    print('Optimal fluxes:', optimal_fluxes)

    return optimal_fluxes


def set_up_inverse_fba(optimal_fluxes: np.ndarray, case: FullObservabilityCase):
    experimental_fluxes_tensor = torch.tensor(optimal_fluxes, dtype=torch.float32)

    D = torch.tensor(np.random.rand(len(optimal_fluxes)), dtype=torch.float32)
    C = torch.tensor(D / torch.norm(D), requires_grad=True)

    print(f"Initial C gradients: {C.grad}")

    fba_layer = setup_optimization_with_basis_functions(case)

    return fba_layer, C, experimental_fluxes_tensor


def plot_loss(loss_history: list, coefficients_history: list, C: torch.Tensor):
    plt.plot([np.log(loss) for loss in loss_history])
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.savefig('loss_history.png')

    coefficients_history = np.array(coefficients_history)
    coefficients = C.detach().cpu().numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(coefficients)), coefficients, color='skyblue')
    plt.xlabel('Flux Indices')
    plt.ylabel('Coefficient Values')
    plt.title('Optimized Coefficients for Measured Fluxes')
    plt.xticks(range(len(coefficients)), range(len(coefficients)))
    plt.tight_layout()
    plt.savefig('coefficients')
    plt.close()


def main(C: torch.Tensor, fba_layer: CvxpyLayer, experimental_fluxes_tensor: torch.Tensor, lr: float,  num_epochs: int,
         optim_option: str):
    loss_history = []
    coefficients_history = []
    loss_fn = torch.nn.functional.mse_loss

    if optim_option == 'LBFGS':
        optimizer = torch.optim.LBFGS([C], lr=lr, max_iter=20)
    else:
        optimizer = torch.optim.SGD([C], lr=lr)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-7
    )

    for epoch in range(num_epochs):
        if optim_option == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                predicted_fluxes, = fba_layer(C)
                loss = loss_fn(predicted_fluxes, experimental_fluxes_tensor)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure().item()
        else:
            optimizer.zero_grad()
            predicted_fluxes, = fba_layer(C)
            loss = loss_fn(predicted_fluxes, experimental_fluxes_tensor)
            loss.backward()
            optimizer.step()

        # Learning rate scheduler step
        scheduler.step(loss)

        loss_history.append(loss)
        coefficients_history.append(C.clone().detach().cpu().numpy())
        print(f'Epoch {epoch}, Loss: {loss}')

        # Optional early stopping on min_lr
        if optimizer.param_groups[0]['lr'] <= 1e-7:
            print("📉 Learning rate reached minimum threshold. Stopping early.")
            break

    return loss_history, C


def multi_start_inverse_fba(case: FullObservabilityCase, optimal_fluxes: np.ndarray, num_starts: int = 5) -> Tuple[torch.Tensor, CvxpyLayer]:
    best_loss = float('inf')
    best_C = None
    best_layer = None

    experimental_fluxes_tensor = torch.tensor(optimal_fluxes, dtype=torch.float32)

    for i in range(num_starts):
        print(f"\n🔁 Multi-start round {i+1}/{num_starts}")

        # Random unit vector
        D = torch.randn(len(optimal_fluxes))
        C = torch.tensor(D / torch.norm(D), dtype=torch.float32, requires_grad=True)

        # Setup optimization problem fresh
        fba_layer = setup_optimization_with_basis_functions(case)

        # Run optimization
        final_loss = main(C, fba_layer, experimental_fluxes_tensor, lr=1e-3, num_epochs=200, optim_option='LBFGS')

        print(f"   ⤷ Final loss: {final_loss}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_C = C.detach().clone()
            best_layer = fba_layer

    print(f"\n✅ Best loss across {num_starts} runs: {best_loss}")
    return best_C, best_layer



def solve_forward_pass(C: torch.Tensor, fba_layer: CvxpyLayer):
    predicted_fluxes, = fba_layer(C)
    print('Predicted fluxes:', predicted_fluxes)




def check_kkt_conditions(C: torch.Tensor, case: FullObservabilityCase, mode: str):
    import numpy as np, cvxpy as cp

    # --- Pull to numpy
    C_np = C.detach().cpu().numpy().astype(float)
    S_np = case.S.detach().cpu().numpy().astype(float) if torch.is_tensor(case.S) else np.asarray(case.S, float)
    vmin = case.v_min.detach().cpu().numpy().astype(float) if torch.is_tensor(case.v_min) else np.asarray(case.v_min, float)
    vmax = case.v_max.detach().cpu().numpy().astype(float) if torch.is_tensor(case.v_max) else np.asarray(case.v_max, float)
    n = case.num_fluxes
    eps = 1e-6

    # --- Rebuild problem in <= form (to get clean dual signs)
    v = cp.Variable(n)
    Cpar = cp.Parameter(n)
    z = cp.multiply(Cpar, v)

    if mode == "norm":
        obj = cp.Minimize(cp.norm(z, 2) + eps)
        sgn = +1
    elif mode == "sum_squares":
        obj = cp.Minimize(cp.sum_squares(z))
        sgn = +1
    elif mode == "logsumexp":
        obj = cp.Minimize(cp.log_sum_exp(z))
        sgn = +1
    elif mode == "linear":
        obj = cp.Maximize(cp.sum(z))
        sgn = -1
    elif mode == "inverse_log":
        obj = cp.Maximize(cp.log(cp.sum(z) + eps))
        sgn = -1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cons = [
        S_np @ v == 0,
        v - vmax <= 0,       # λ_upper ≥ 0
        vmin - v <= 0        # λ_lower ≥ 0
    ]
    prob = cp.Problem(obj, cons)
    Cpar.value = C_np
    prob.solve(solver=cp.SCS, verbose=False)

    v_sol = v.value
    if v_sol is None:
        return {"error": "solver_failed"}

    mu        = np.asarray(cons[0].dual_value, float).ravel()
    lam_upper = np.asarray(cons[1].dual_value, float).ravel()   # ≥ 0
    lam_lower = np.asarray(cons[2].dual_value, float).ravel()   # ≥ 0

    # --- Primal feasibility
    eq_res = np.linalg.norm(S_np @ v_sol)
    #bounds_ok = np.all(v_sol <= vmax + 1e-9) and np.all(v_sol >= vmin - 1e-9)
    def bounds_ok_with_tol(v, vmin, vmax, tol=1e-7):
        upper_viol = np.max(v - vmax)
        lower_viol = np.max(vmin - v)
        ok = (upper_viol <= tol) and (lower_viol <= tol)
        return ok, float(max(upper_viol, 0.0)), float(max(lower_viol, 0.0))

    bounds_ok, up_violation, low_violation = bounds_ok_with_tol(v_sol, vmin, vmax, tol=1e-7)

    # --- Gradient wrt v (mode-specific)
    z_np = C_np * v_sol
    if mode == "norm":
        nz = np.linalg.norm(z_np)
        if nz > 0:
            grad_f = (C_np**2) * v_sol / nz
            grad_defined = True
        else:
            grad_f = np.zeros_like(v_sol)  # subgradient set-valued
            grad_defined = False
    elif mode == "sum_squares":
        grad_f = 2.0 * (C_np**2) * v_sol
        grad_defined = True
    elif mode == "logsumexp":
        # softmax(z)
        m = np.max(z_np)
        sm = np.exp(z_np - m)
        sm /= np.sum(sm)
        grad_f = sm * C_np
        grad_defined = True
    elif mode == "linear":
        grad_f = C_np
        grad_defined = True
    elif mode == "inverse_log":
        s = np.sum(z_np) + eps
        if s <= 0:
            return {"error": "domain_error_inverse_log", "sum_Cv": float(s - eps)}
        grad_f = C_np / s
        grad_defined = True

    # --- Stationarity:  sgn*∇f + S^T μ + λ_upper − λ_lower = 0
    stationarity_vec = sgn * grad_f + S_np.T @ mu + lam_upper - lam_lower
    stationarity_res = np.linalg.norm(stationarity_vec)

    # --- Complementary slackness
    comp_upper = np.max(np.abs(lam_upper * (v_sol - vmax)))
    comp_lower = np.max(np.abs(lam_lower * (vmin - v_sol)))

    return {
        "primal_feasibility": float(eq_res),
        "bounds_feasible": bool(bounds_ok),
        "max_upper_violation": up_violation,
        "max_lower_violation": low_violation,
        "stationarity_residual": float(stationarity_res),
        "comp_upper": float(comp_upper),
        "comp_lower": float(comp_lower),
        "grad_defined": bool(grad_defined),
        "objective_value": float(prob.value),
        "duals_norms": {
            "mu": float(np.linalg.norm(mu)),
            "lam_upper": float(np.linalg.norm(lam_upper)),
            "lam_lower": float(np.linalg.norm(lam_lower)),
        }
    }


def old_check_kkt_conditions(C: torch.Tensor, case: FullObservabilityCase, mode):
    C_numpy = C.detach().cpu().numpy()
    S_np = case.S.detach().cpu().numpy() if torch.is_tensor(case.S) else case.S

    # Rebuild the same CVXPY problem (outside cvxpylayer) to extract duals
    v = cp.Variable(case.num_fluxes)
    C_param = cp.Parameter(case.num_fluxes)

    objective = cp.Minimize(cp.norm(cp.multiply(C_param, v)))
    constraints = [case.S @ v == 0, v >= case.v_min, v <= case.v_max]
    problem = cp.Problem(objective, constraints)

    C_param.value = C_numpy
    problem.solve(solver=cp.ECOS, verbose=False)

    v_sol = v.value
    mu = constraints[0].dual_value  # duals for Sv = 0
    lambda_min = constraints[1].dual_value  # duals for v >= v_min
    lambda_max = constraints[2].dual_value  # duals for v <= v_max

    # Early exit if any duals are missing
    if mu is None or lambda_min is None or lambda_max is None:
        print("❌ Cannot verify KKT conditions: some dual variables are missing (None).")
        return

    print("Shapes:")
    print("  S.T @ mu:", (case.S.T @ mu).shape)
    print("  lambda_min:", lambda_min.shape)
    print("  lambda_max:", lambda_max.shape)

    # flat 1D array
    mu = np.array(mu).flatten()
    lambda_min = np.array(lambda_min).flatten()
    lambda_max = np.array(lambda_max).flatten()

    print("🔍 Primal feasibility (Sv):", np.linalg.norm(case.S @ v_sol))
    v_min_np = case.v_min.detach().cpu().numpy() if torch.is_tensor(case.v_min) else np.array(case.v_min)
    v_max_np = case.v_max.detach().cpu().numpy() if torch.is_tensor(case.v_max) else np.array(case.v_max)

    print("🔍 v within bounds:", np.all(v_sol >= v_min_np), np.all(v_sol <= v_max_np))

    grad_f = C_numpy ** 2 * v_sol / np.linalg.norm(C_numpy * v_sol)
    kkt_residual = grad_f + S_np.T @ mu + lambda_min - lambda_max

    print("📐 Stationarity residual (∇L = 0):", np.linalg.norm(kkt_residual))

    comp_slack_min = lambda_min * (v_sol - v_min_np)
    comp_slack_max = lambda_max * (v_max_np - v_sol)
    print("🔁 Complementary slackness residuals:")
    print("   min bound:", np.max(np.abs(comp_slack_min)))
    print("   max bound:", np.max(np.abs(comp_slack_max)))

    return {
        "primal_feasibility": np.linalg.norm(S_np @ v_sol),
        "bounds_feasible": np.all(v_sol >= v_min_np) and np.all(v_sol <= v_max_np),
        "stationarity_residual": np.linalg.norm(kkt_residual),
        "comp_slack_min": np.max(np.abs(comp_slack_min)),
        "comp_slack_max": np.max(np.abs(comp_slack_max))
    }


def interpret_kkt(
    residuals: dict,
    tol_primal: float = 1e-7,
    tol_bounds: float = 1e-7,
    tol_stat_abs: float = 1e-5,
    tol_stat_rel: float = 1e-6,
    tol_comp: float = 1e-6,
):
    if residuals is None:
        print("❌ No KKT data.")
        return

    if "error" in residuals:
        print(f"❌ KKT check aborted: {residuals['error']}")
        if "sum_Cv" in residuals:
            print(f"   sum(C⊙v): {residuals['sum_Cv']:.3e}")
        return

    # Backward compatibility for key names
    comp_lower = float(residuals.get("comp_lower",
                      residuals.get("comp_slack_min", float("inf"))))
    comp_upper = float(residuals.get("comp_upper",
                      residuals.get("comp_slack_max", float("inf"))))
    stat_res   = float(residuals.get("stationarity_residual", float("inf")))
    prim_res   = float(residuals.get("primal_feasibility", float("inf")))
    bounds_ok  = bool(residuals.get("bounds_feasible", False))
    grad_def   = bool(residuals.get("grad_defined", True))

    max_up_viol  = float(residuals.get("max_upper_violation", 0.0))
    max_low_viol = float(residuals.get("max_lower_violation", 0.0))

    # Optional relative stationarity gate
    obj_val = float(residuals.get("objective_value", 0.0))
    scale = 1.0 + abs(obj_val)  # simple scale proxy; use ||grad|| if pass it through residuals
    stat_ok = (stat_res < tol_stat_abs) or ((stat_res / scale) < tol_stat_rel)

    print("🔍 KKT Summary")
    print(f"   ‣ Primal equality ‖S v‖:         {prim_res:.3e}   → {'OK' if prim_res < tol_primal else 'FAIL'}")
    print(f"   ‣ Bounds feasible:                {bounds_ok} "
          f"(max↑viol={max_up_viol:.3e}, max↓viol={max_low_viol:.3e})")
    print(f"   ‣ Stationarity ‖∇L‖:             {stat_res:.3e}   "
          f"→ {'OK' if stat_ok else 'FAIL'} "
          f"[abs<{tol_stat_abs:.1e} or rel<{tol_stat_rel:.1e}]")
    print(f"   ‣ Complementarity (lower):        {comp_lower:.3e} → {'OK' if comp_lower < tol_comp else 'FAIL'}")
    print(f"   ‣ Complementarity (upper):        {comp_upper:.3e} → {'OK' if comp_upper < tol_comp else 'FAIL'}")

    if not grad_def:
        print("⚠️  Gradient undefined (subgradient case, e.g. ‖diag(C)v‖ = 0). "
              "Stationarity pass/fail may be inconclusive.")

    if "objective_value" in residuals:
        print(f"   ‣ Objective value:                {obj_val:.6g}")

    if "duals_norms" in residuals:
        dn = residuals["duals_norms"]
        mu_n   = float(dn.get("mu", float("nan")))
        up_n   = float(dn.get("lam_upper", float("nan")))
        low_n  = float(dn.get("lam_lower", float("nan")))
        print(f"   ‣ ‖μ‖: {mu_n:.3e} | ‖λ_upper‖: {up_n:.3e} | ‖λ_lower‖: {low_n:.3e}")

    all_ok = (
        (prim_res < tol_primal) and
        bounds_ok and (max_up_viol <= tol_bounds) and (max_low_viol <= tol_bounds) and
        stat_ok and
        (comp_lower < tol_comp) and (comp_upper < tol_comp)
    )
    print(f"\n✅ Overall KKT satisfied: {all_ok}")


def original_single_start(case, optimal_fluxes, run_id, mode):
    print(f"🔁 Start {run_id}")
    experimental_fluxes_tensor = torch.tensor(optimal_fluxes, dtype=torch.float32)

    D = torch.randn(len(optimal_fluxes))
    C = torch.tensor(D / torch.norm(D), dtype=torch.float32, requires_grad=True)
    fba_layer = setup_optimization_with_basis_functions(case, mode)

    loss_history, final_C = main(C, fba_layer, experimental_fluxes_tensor, lr=1e-3, num_epochs=100, optim_option='LBFGS')
    final_loss = loss_history[-1]
    return final_loss, final_C.detach().clone(), fba_layer, loss_history


def single_start(case, optimal_fluxes, run_id, mode, lr, num_epochs):
    print(f"🔁 Start {run_id}")
    experimental_fluxes_tensor = torch.tensor(optimal_fluxes, dtype=torch.float32)

    # 🧠 Define the loss wrapper for Bayesian optimization
    def f(C_tensor):
        C_tensor = C_tensor.detach().clone().requires_grad_(True)
        fba_layer = setup_optimization_with_basis_functions(case, mode)
        loss_history, _ = main(C_tensor, fba_layer, experimental_fluxes_tensor, lr=lr, num_epochs=15, optim_option='LBFGS')
        return torch.tensor([loss_history[-1]])

    # # 🔢 Bounds: [number of measured flux indices + 1]
    # # if regularization +1
    # dim = len(optimal_fluxes)
    #
    # lower_bound = torch.zeros(dim)
    # upper_bound = torch.ones(dim)
    #
    # # 🔍 Bayesian optimization to find a good initial C
    # best_candidate = bayesian_optimization(
    #     f=f,
    #     lower_bound=lower_bound,
    #     upper_bound=upper_bound,
    #     num_initial_points=10,
    #     num_iterations=1  # Increase for deeper search
    # )
    #
    # # 🎯 Use best_candidate as C
    # C = best_candidate.clone().detach().requires_grad_(True)

    # randomly start between 0 and 1
    C = torch.rand(len(optimal_fluxes), dtype=torch.float32, requires_grad=True)

    # 🔁 Now do the full optimization
    fba_layer = setup_optimization_with_basis_functions(case, mode)
    loss_history, final_C = main(C, fba_layer, experimental_fluxes_tensor, lr=1e-3, num_epochs=num_epochs, optim_option='LBFGS')
    final_loss = loss_history[-1]

    return final_loss, final_C.detach().clone(), fba_layer, loss_history



def multi_start_inverse_fba_parallel(case: FullObservabilityCase, optimal_fluxes: np.ndarray, num_starts: int, mode: str, lr: float, num_epochs: int):
    with parallel_backend("threading"):
        results = Parallel(n_jobs=num_starts)(
            delayed(single_start)(case, optimal_fluxes, i, mode, lr, num_epochs) for i in range(num_starts)
        )

    # Now include loss history
    best_result = min(results, key=lambda x: x[0])  # pick by final loss
    best_loss, best_C, best_layer, best_loss_history = best_result
    return best_C, best_layer, best_loss_history

def plot_summary(C: torch.Tensor, predicted_fluxes: torch.Tensor, true_fluxes: torch.Tensor, mode: str, output_dir):

    coeffs = C.detach().cpu().numpy().flatten()
    predicted = predicted_fluxes.detach().cpu().numpy()
    true = true_fluxes.detach().cpu().numpy()

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Bar plot of coefficients
    plt.figure(figsize=(7, 5))
    plt.bar(range(len(coeffs)), coeffs, color='skyblue')
    plt.xlabel('Flux Index')
    plt.ylabel('Coefficient Value')
    plt.title(f'Optimized Coefficients ({mode})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'coefficients_{mode}.pdf'), format='pdf')
    plt.close()

    # Predicted vs measured fluxes
    plt.figure(figsize=(7, 5))
    plt.plot(true, 'o-', label='Measured')
    plt.plot(predicted, 'x--', label='Predicted')
    plt.xlabel('Flux Index')
    plt.ylabel('Flux Value')
    plt.title(f'Flux Comparison ({mode})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'flux_comparison_{mode}.pdf'), format='pdf')
    plt.close()


def save_flux_and_coefficients(C: torch.Tensor, predicted_fluxes: torch.Tensor, true_fluxes: torch.Tensor, mode: str):
    output_dir = os.path.join("ThesisResults", "full_obs", mode)
    os.makedirs(output_dir, exist_ok=True)

    coeffs = C.detach().cpu().numpy().flatten()
    predicted = predicted_fluxes.detach().cpu().numpy()
    true = true_fluxes.detach().cpu().numpy()

    with open(os.path.join(output_dir, "flux_summary.txt"), "w") as f:
        f.write(f"# Convexification mode: {mode}\n\n")
        f.write("Experimental fluxes:\n")
        f.write(" ".join([f"{x:.6f}" for x in true]) + "\n\n")

        f.write("Predicted fluxes:\n")
        f.write(" ".join([f"{x:.6f}" for x in predicted]) + "\n\n")

        f.write("Optimized coefficients:\n")
        f.write(" ".join([f"{x:.6f}" for x in coeffs]) + "\n")


import csv

def init_kkt_csv():
    csv_path = os.path.join("ThesisResults", "full_obs", "kkt_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "primal_feasibility", "bounds_feasible",
                         "stationarity_residual", "comp_slack_min", "comp_slack_max"])

import os, csv, math
from datetime import datetime

def append_kkt_csv(residuals: dict, mode: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "kkt_summary.csv")

    # Back-compat & safe gets
    prim   = float(residuals.get("primal_feasibility", math.nan))
    b_ok   = bool(residuals.get("bounds_feasible", False))
    stat   = float(residuals.get("stationarity_residual", math.nan))
    comp_l = float(residuals.get("comp_lower", residuals.get("comp_slack_min", math.nan)))
    comp_u = float(residuals.get("comp_upper", residuals.get("comp_slack_max", math.nan)))
    up_vio = float(residuals.get("max_upper_violation", math.nan))
    lo_vio = float(residuals.get("max_lower_violation", math.nan))
    obj    = float(residuals.get("objective_value", math.nan))
    gd     = bool(residuals.get("grad_defined", True))

    dn = residuals.get("duals_norms", {}) or {}
    mu_n  = float(dn.get("mu", math.nan))
    lup_n = float(dn.get("lam_upper", math.nan))
    llo_n = float(dn.get("lam_lower", math.nan))

    row = [
        datetime.utcnow().isoformat(),  # timestamp (optional but handy)
        mode,
        f"{prim:.6e}",
        int(b_ok),
        f"{up_vio:.6e}",
        f"{lo_vio:.6e}",
        f"{stat:.6e}",
        f"{comp_l:.6e}",
        f"{comp_u:.6e}",
        f"{obj:.6e}",
        f"{mu_n:.6e}",
        f"{lup_n:.6e}",
        f"{llo_n:.6e}",
        int(gd),
    ]

    header = [
        "timestamp_utc",
        "mode",
        "primal_feasibility",
        "bounds_feasible",
        "max_upper_violation",
        "max_lower_violation",
        "stationarity_residual",
        "comp_lower",
        "comp_upper",
        "objective_value",
        "norm_mu",
        "norm_lambda_upper",
        "norm_lambda_lower",
        "grad_defined",
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def plot_loss_history(loss_history: list, mode: str):
    import os
    output_dir = os.path.join("ThesisResults", "full_obs", mode)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(loss_history, label='Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Loss History ({mode})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_history_{mode}.pdf"), format='pdf')
    plt.close()


if __name__ == '__main__':

    geometric_fba = FullObservabilityCase(
        num_fluxes=5,
        v_min=[-1, -10000, 0, -2, 0],
        v_max=[1, -2, 2, 10000, 10000],
        S=np.array([
            [1, -1, -1, -1, 0],
            [0, 1, 1, 1, -1],
        ])
    )

    # geometric_fba= FullObservabilityCase(
    #     num_fluxes=5,
    #     v_min=[-1, 0, 0, 0, 0],
    #     v_max=[1, 2, 2, 10000, 10000],
    #     S=np.array([
    #         [1, 0, 0, 0, 0],
    #         [0, 1, 1, 1, -1],
    #     ])
    # )

    optimal_fluxes = setup_ground_truth(geometric_fba)

    init_kkt_csv()

    for mode in ["logsumexp","norm","sum_squares"]:
        print(f"\n🔍 Convexification: {mode.upper()}")
        try:
            best_C, fba_layer, loss_history  = multi_start_inverse_fba_parallel(geometric_fba, optimal_fluxes, num_starts=1, mode=mode, lr=1e-3, num_epochs=50)
            predicted_fluxes, = fba_layer(best_C)
            measured_fluxes = torch.tensor(optimal_fluxes, dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(predicted_fluxes, torch.tensor(optimal_fluxes, dtype=torch.float32)).item()
            print(f"Loss: {loss:.4f}")

            # Save loss history to txt
            output_dir = os.path.join("ThesisResults_V6", "full_obs_random_SCS", mode)
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, "loss_history.txt"), "w") as f:
                for i, l in enumerate(loss_history):
                    f.write(f"{i}\t{l:.6f}\n")

            # 🔹 1. Save loss history
            loss_df = pd.DataFrame({"epoch": list(range(len(loss_history))), "loss": loss_history})
            loss_df.to_csv(os.path.join(output_dir, "loss_history.csv"), index=False)

            # 🔹 2. Save final coefficients
            C_np = best_C.detach().cpu().numpy()
            pd.DataFrame({"C": C_np}).to_csv(os.path.join(output_dir, "C_final.csv"), index=False)

            # 🔹 3. Save predicted vs measured fluxes
            flux_df = pd.DataFrame({
                "flux_index": list(range(len(predicted_fluxes))),
                "measured": measured_fluxes.detach().cpu().numpy(),
                "predicted": predicted_fluxes.detach().cpu().numpy()
            })
            flux_df.to_csv(os.path.join(output_dir, "flux_comparison.csv"), index=False)

            plot_loss_summary = True
            if plot_loss_summary:
                plot_loss_history(loss_history, mode)

            plot_summary(best_C, predicted_fluxes, torch.tensor(optimal_fluxes, dtype=torch.float32), mode, output_dir)

            residuals = check_kkt_conditions(best_C, geometric_fba, mode)
            print(" Final KKT residuals:", residuals)
            interpret_kkt(residuals)
            append_kkt_csv(residuals, mode, output_dir)
        except Exception as e:
            print(f" Mode {mode.upper()} failed with error: {e}")
            continue

