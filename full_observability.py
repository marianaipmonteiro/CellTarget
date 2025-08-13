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


def check_kkt_conditions(C: torch.Tensor, case: FullObservabilityCase, mode):
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


def interpret_kkt(residuals, tol=1e-5):
    print("🔍 KKT Summary:")
    print("   ✅ Primal feasibility:", residuals["primal_feasibility"] < tol)
    print("   ✅ Bounds feasible:", residuals["bounds_feasible"])
    print("   ✅ Stationarity OK:", residuals["stationarity_residual"] < tol)
    print("   ✅ Slack min OK:", residuals["comp_slack_min"] < tol)
    print("   ✅ Slack max OK:", residuals["comp_slack_max"] < tol)


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

    # 🔢 Bounds: [number of measured flux indices + 1]
    # if regularization +1
    dim = len(optimal_fluxes)

    lower_bound = torch.zeros(dim)
    upper_bound = torch.ones(dim)

    # 🔍 Bayesian optimization to find a good initial C
    best_candidate = bayesian_optimization(
        f=f,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_initial_points=10,
        num_iterations=1  # Increase if you want deeper search
    )

    # 🎯 Use best_candidate as C
    C = best_candidate.clone().detach().requires_grad_(True)

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
    output_dir = os.path.join("Publication_Results", "full_obs", mode)
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
    csv_path = os.path.join("Publication_Results", "full_obs", "kkt_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "primal_feasibility", "bounds_feasible",
                         "stationarity_residual", "comp_slack_min", "comp_slack_max"])

def append_kkt_csv(residuals: dict, mode: str, output_dir):
    csv_path = os.path.join(output_dir, "kkt_summary.csv")
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            mode,
            f"{residuals['primal_feasibility']:.6e}",
            int(residuals['bounds_feasible']),
            f"{residuals['stationarity_residual']:.6e}",
            f"{residuals['comp_slack_min']:.6e}",
            f"{residuals['comp_slack_max']:.6e}",
        ])

def plot_loss_history(loss_history: list, mode: str):
    import os
    output_dir = os.path.join("Publication_Results", "full_obs", mode)
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


    optimal_fluxes = setup_ground_truth(geometric_fba)

    init_kkt_csv()

    for mode in ["logsumexp","norm","sum_squares"]:
        print(f"\n🔍 Convexification: {mode.upper()}")
        try:
            best_C, fba_layer, loss_history  = multi_start_inverse_fba_parallel(geometric_fba, optimal_fluxes, num_starts=5, mode=mode, lr=1e-3, num_epochs=200)
            predicted_fluxes, = fba_layer(best_C)
            measured_fluxes = torch.tensor(optimal_fluxes, dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(predicted_fluxes, torch.tensor(optimal_fluxes, dtype=torch.float32)).item()
            print(f"Loss: {loss:.4f}")

            # Save loss history to txt
            output_dir = os.path.join("Publication_Results", "full_obs", mode)
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
            print("✅ Final KKT residuals:", residuals)
            interpret_kkt(residuals)
            append_kkt_csv(residuals, mode, output_dir)
        except Exception as e:
            print(f"❌ Mode {mode.upper()} failed with error: {e}")
            continue

