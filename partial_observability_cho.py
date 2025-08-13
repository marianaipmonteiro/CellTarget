import argparse
import logging
import os
from typing import List, Tuple, Callable, Sequence, Dict, Any
import cvxpylayers
import pandas as pd
import cobra
from cobra.io import load_json_model
import numpy as np
import cvxpy as cp
import torch
from joblib import parallel_backend, Parallel, delayed
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for workers
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F
from functools import partial
from bo import bayesian_optimization
from setup import PartialObservabilityCase
import csv





def plot_grad_per_dimension(grad_dir_history: torch.Tensor, mode: str, labels, output_dir: str, logy: bool = True):
    """
    grad_dir_history: [T, m] tensor of per-epoch gradient norms per measured flux direction.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)
    G = grad_dir_history.cpu().numpy()
    T, m = G.shape

    plt.figure(figsize=(7,4))
    for j in range(m):
        name = labels[j] if labels is not None and j < len(labels) else f"dir {j}"
        plt.plot(G[:, j], label=name, linewidth=1.5)
    if logy:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("|| gradient contribution ||")
    plt.title(f"Per-dimension gradient magnitude ({mode})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"grad_per_dimension_{mode}.png"), dpi=200)
    plt.close()

    # print loss per dimension to a csv
    with open(os.path.join(output_dir, f"grad_per_dimension_{mode}.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch"] + [f"dir_{j}" for j in range(m)])
        for t in range(T):
            writer.writerow([t] + list(G[t, :]))


def plot_loss_history(loss_history: list, mode: str,  output_dir: str):
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

    # print loss history to a csv
    with open(os.path.join(output_dir, f"loss_history_{mode}.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Loss"])
        for t, loss in enumerate(loss_history):
            writer.writerow([t, loss])


def plot_summary(C: torch.Tensor, predicted_fluxes: torch.Tensor, true_fluxes: torch.Tensor, mode: str,  output_dir:str, objective_fluxes):
    os.makedirs(output_dir, exist_ok=True)

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


    x = np.arange(len(objective_fluxes))

    plt.figure(figsize=(8, 5))
    plt.scatter(x, true, label='Measured', marker='o')
    plt.scatter(x, predicted, label='Predicted', marker='x')
    plt.xticks(x, objective_fluxes, rotation=45, ha='right')
    plt.xlabel('Flux')
    plt.ylabel('Flux Value')
    plt.title(f'Flux Comparison ({mode})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'flux_comparison_{mode}.pdf'), format='pdf')
    plt.close()

def save_flux_and_coefficients(C: torch.Tensor, predicted_fluxes: torch.Tensor, true_fluxes: torch.Tensor, mode: str, output_dir: str):
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




def optimize_C_partial_lbfgs(
    C: torch.Tensor,                                          # (in) shape: [k], dtype: float32
    fba_layer: Callable[[torch.Tensor], Tuple[torch.Tensor]], # (in) maps C -> (predicted_fluxes,)
    experimental_fluxes_tensor: torch.Tensor,                 # (in) shape: [n], dtype: float32
    measured_flux_indices: Sequence[int],                     # (in) indices into flux vector
    lr: float,                                                # (in) LBFGS step size (often ~1.0)
    num_epochs: int,                                          # (in) outer iterations (LBFGS steps)
    lambda_reg: float,                                        # (in) L2 weight on C
) -> Tuple[List[float], torch.Tensor, dict]:                        # (out) (loss_history, optimized C)
    C = torch.nn.Parameter(C.detach().clone(), requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [C],
        lr=lr,
        max_iter=20,              # per-step inner iters (tweak)
        max_eval=25,              # evaluations cap (tweak)
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    flux_subset = experimental_fluxes_tensor[measured_flux_indices]
    loss_history: List[float] = []
    grad_dir_history: List[List[float]] = []  # will be [T, m]

    # pure eval (no backward) to measure loss at current C
    @torch.no_grad()
    def eval_loss_only() -> float:
        (pred,) = fba_layer(C)
        pred_sub = pred[measured_flux_indices]
        loss_mse = torch.nn.functional.mse_loss(pred_sub, flux_subset)
        loss = loss_mse + lambda_reg * torch.sum(C ** 2)
        return float(loss)

    # init best trackers (at starting C)
    best_loss = eval_loss_only()
    best_C = C.detach().clone()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        (predicted_fluxes,) = fba_layer(C)
        predicted_subset = predicted_fluxes[measured_flux_indices]
        loss = torch.nn.functional.mse_loss(predicted_subset, flux_subset)
        loss = loss + lambda_reg * torch.sum(C ** 2)
        loss.backward()
        return loss

    m = len(measured_flux_indices)

    for _ in range(num_epochs):
        # ----  compute per-dimension gradient contributions (MSE part) ----
        optimizer.zero_grad(set_to_none=True)
        (predicted_fluxes,) = fba_layer(C)
        predicted_subset = predicted_fluxes[measured_flux_indices]
        residual = predicted_subset - flux_subset  # [m]

        per_dim_norms = []
        for i in range(m):
            unit = torch.zeros_like(predicted_subset)
            unit[i] = 1.0
            # dpred_i/dC
            Ji = torch.autograd.grad(
                outputs=predicted_subset,
                inputs=C,
                grad_outputs=unit,
                retain_graph=True,
                create_graph=False,
                allow_unused=False
            )[0]  # [k]
            # Gradient contribution of dimension i to d/dC (MSE term):
            # (2/m)*r_i * dpred_i/dC
            gi = (2.0 / m) * residual[i] * Ji
            per_dim_norms.append(gi.norm().item())

        grad_dir_history.append(per_dim_norms)
        # ---------------------------------------------------------------------

        loss_val = closure()
        grad_norm = C.grad.norm().item() if C.grad is not None else float("nan")
        optimizer.step(closure)   # may call closure multiple times internally

        # Record post-step loss (actual new parameters)
        post_step_loss = eval_loss_only()
        loss_history.append(post_step_loss)
        if post_step_loss < best_loss:
            best_loss = post_step_loss
            best_C = C.detach().clone()
        print(f"Loss: {loss_history[-1]:.6e} | ∇C: {grad_norm:.3e}")

    extras = {
        "grad_dir_history": torch.tensor(grad_dir_history)  # [T, m]
    }
    return loss_history, best_C , extras


def optimize_C_partial(
    C,
    fba_layer,
    experimental_fluxes_tensor,
    measured_flux_indices,
    lr,
    num_epochs,
    lambda_reg,
    use_scheduler=True
):
    C = torch.nn.Parameter(C.detach().clone(), requires_grad=True)
    optimizer = torch.optim.Adam([C], lr=lr)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(1e-8 / lr) ** (1 / num_epochs)
        )

    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predicted_fluxes, = fba_layer(C)

        flux_subset = experimental_fluxes_tensor[measured_flux_indices]
        predicted_subset = predicted_fluxes[measured_flux_indices]

        loss = torch.nn.functional.mse_loss(predicted_subset, flux_subset)
        loss += lambda_reg * torch.sum(C**2)

        loss.backward()
        optimizer.step()
        if use_scheduler:
            scheduler.step()

        loss_history.append(loss.item())
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | ∇C: {C.grad.norm().item():.4e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    return loss_history, C


def single_start_partial(
    run_id,
    mode,
    fba_layer,
    experimental_fluxes_tensor,
    measured_flux_indices,
    output_dir,
    num_epochs,
    lambda_reg,
):
    print(f"🔁 Partial Start {run_id}")

    run_dir = os.path.join(output_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # === BO initialization function ===
    def f(C_tensor):
        C_tensor = C_tensor.detach().clone().requires_grad_(True)
        loss_history, _, _ = optimize_C_partial_lbfgs(
            C_tensor,
            fba_layer,
            experimental_fluxes_tensor,
            measured_flux_indices,
            lr=1,
            num_epochs=10,
            lambda_reg=lambda_reg
        )
        return torch.tensor([loss_history[-1]])

    dim = len(measured_flux_indices)
    lower_bound = torch.zeros(dim)
    upper_bound = torch.ones(dim)

    best_candidate = bayesian_optimization(
        f=f,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_initial_points=5,
        num_iterations=10
    )

    # === Final training ===
    C = best_candidate.clone().detach().requires_grad_(True)

    loss_history, final_C, extras = optimize_C_partial_lbfgs(
        C,
        fba_layer,
        experimental_fluxes_tensor,
        measured_flux_indices,
        lr=1,
        num_epochs=num_epochs,
        lambda_reg=lambda_reg
    )

    final_loss: float = float(loss_history[-1])

    measured_labels = [f"flux_{i}" for i in range(len(measured_flux_indices))]  # or your real names

    # Compute predictions once
    predicted_fluxes, = fba_layer(final_C.detach())
    predicted_subset = predicted_fluxes[measured_flux_indices]
    experimental_subset = experimental_fluxes_tensor[measured_flux_indices]

    # Save plots/files per run
    plot_grad_per_dimension(extras["grad_dir_history"], mode, [f"flux_{i}" for i in range(len(measured_flux_indices))], run_dir, logy=True)
    plot_summary(final_C.detach(), predicted_subset, experimental_subset, mode, run_dir, measured_labels)
    save_flux_and_coefficients(final_C.detach(), predicted_subset, experimental_subset, mode, run_dir)
    plot_loss_history(loss_history, mode, run_dir)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "loss_history": loss_history,
        "predicted_subset": predicted_subset.detach().cpu(),
        "experimental_subset": experimental_subset.detach().cpu(),
        "run_dir": run_dir,
        "mode": mode,
    }
    return final_loss, final_C.detach().clone(), summary




def multi_start_inverse_fba_partial_parallel(
    fba_layer,
    experimental_fluxes_tensor,
    measured_flux_indices,
    output_dir,
    num_starts: int,
    mode: str,
    num_epochs: int,
    lambda_reg: float
):
    with parallel_backend("threading"):
        results = Parallel(n_jobs=num_starts)(
            delayed(single_start_partial)(
                run_id=i,
                mode=mode,
                fba_layer=fba_layer,
                experimental_fluxes_tensor=experimental_fluxes_tensor,
                measured_flux_indices=measured_flux_indices,
                output_dir=output_dir,
                num_epochs=num_epochs,
                lambda_reg=lambda_reg
            )
            for i in range(num_starts)
        )

    # results elements are now: (final_loss, final_C, summary)
    best_result = min(results, key=lambda x: x[0])
    best_loss, best_C, best_summary = best_result

    # keep old return contract (best_C, best_layer, best_loss_history)
    best_layer = fba_layer  # same layer for all runs
    best_loss_history = best_summary["loss_history"]

    return best_C, best_layer, best_loss_history


def setup_optimization_with_basis_functions_partial(
    measured_flux_indices: List[int],
    S: np.ndarray,
    v_min: torch.Tensor,
    v_max: torch.Tensor,
    num_reactions: int,
    mode: str
) -> Tuple[CvxpyLayer, cp.Parameter]:
    v = cp.Variable(num_reactions)
    C_param = cp.Parameter(len(measured_flux_indices))
    v_measured = cp.hstack([v[i] for i in measured_flux_indices])

    objective_fitness = cp.multiply(C_param, v_measured)

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
    elif mode == "inv_linear":
        objective_expr = -1.0 / (cp.sum(objective_fitness) + epsilon)
    elif mode == "entropy":
        objective_expr = -cp.sum(cp.entr(objective_fitness + epsilon))  # concave → OK
    else:
        raise ValueError(f"Unknown mode: {mode}")


    v_min_np = v_min.detach().numpy()
    v_max_np = v_max.detach().numpy()

    constraints = [
        #S @ v == 0,
        S @ v <= 0.001, # NOTE: Relaxation of steady-state condition seems to help
        v >= v_min_np,
        v <= v_max_np
    ]

    if mode in ["norm", "sum_squares", "logsumexp"]:
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    else: #inverse_log linear entropy
        problem = cp.Problem(cp.Maximize(objective_expr), constraints)

    base_layer = CvxpyLayer(problem, parameters=[C_param], variables=[v])

    # Pretty-print the problem
    print("Objective:")
    print(problem.objective)
    print("\nConstraints:")
    for c in problem.constraints:
        print(c)

    return base_layer


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

            rxn.lower_bound = new_lb
            rxn.upper_bound = new_ub

    # Optional: access median values if needed
    median_fluxes = np.array([median[rxn.id] for rxn in model.reactions if rxn.id in median.index])

    return model




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


def select_best_lambda_given_C(
    lambda_grid,
    fba_layer,
    C_init,
    experimental_fluxes_tensor,
    measured_flux_indices,
    num_epochs,
    lr=1,
):
    results = []

    for lam in lambda_grid:
        print(f"🔍 Trying lambda_reg = {lam:.0e}")
        loss_history, _, _ = optimize_C_partial_lbfgs(
            C=C_init.clone().detach(),
            fba_layer=fba_layer,
            experimental_fluxes_tensor=experimental_fluxes_tensor,
            measured_flux_indices=measured_flux_indices,
            lr=1,
            num_epochs=num_epochs,
            lambda_reg=lam
        )
        final_loss = loss_history[-1]
        results.append((lam, final_loss))
        print(f"✅ Lambda {lam:.0e} → Final Loss: {final_loss:.6f}")

    best_lambda, best_loss = min(results, key=lambda x: x[1])
    print(f"\n🏆 Best lambda: {best_lambda:.0e} with loss {best_loss:.4f}")
    return best_lambda



if '__main__' == __name__:

    parser = argparse.ArgumentParser(description="Starting inverse FBA.")
    parser.add_argument("--condition", type=str,  help="Path to the input Excel file.")
    parser.add_argument("--output_dir", type=str,  help="Path to the input Excel file.")
    args = parser.parse_args()


    file_path = 'inputs/C13_data.xlsx'
    c13 = pd.read_excel(file_path, sheet_name="c13")
    condition = args.condition #'early' # stat, AWZ11expo, AWZ11stat
    arg_output_dir = args.output_dir

    if condition in ['AWZ11expo', 'AWZ11stat']:
        model = load_model('inputs/CHOmpact_cobrapy_original_nonprod.json')
        objective_fluxes = ['F142', 'F105', 'F107', 'F112', 'F114']    #growth mab glc lac asn gln
    elif condition in ['early', 'stat']:
        model = load_model('inputs/CHOmpact_cobrapy_original.json')
        objective_fluxes = ['F142', 'F143', 'F105', 'F107', 'F112', 'F114']    #growth mab glc lac asn gln
    else:
        raise ValueError(f"Unknown condition: {condition}. Expected one of ['early', 'late', 'AWZ11expo', 'AWZ11stat'].")

    lb_row, ub_row, median_row = extract_bounds(c13, model, condition)
    constrained_model = constrain_model(model, lb_row, ub_row, median_row)
    median_tensor = torch.tensor(median_row.values, dtype=torch.float32)
    output_dir_input = os.path.join(arg_output_dir,condition)

    num_reactions = len(model.reactions)

    S, vmin, vmax, S_torch, v_min_torch, v_max_torch = initialize_S_and_vs(
        model, lb_row, ub_row
    )
    flux_to_index = {rxn.id: i for i, rxn in enumerate(model.reactions)}

    objective_fluxes_indices = [flux_to_index[name] for name in objective_fluxes]
    lambda_grid = [1e-6, 1e-5, 1e-4]

    for mode in ["inverse_log"]: #, "linear", "norm", "sum_squares", "logsumexp"
        output_dir = os.path.join(output_dir_input, mode)
        os.makedirs(output_dir, exist_ok=True)

        fba_layer = setup_optimization_with_basis_functions_partial(
            objective_fluxes_indices, S, v_min_torch, v_max_torch, num_reactions, mode
        )
        # === Step 5.1: Random C for lambda tuning ===
        C_random = torch.rand(len(objective_fluxes_indices), dtype=torch.float32)

        best_lambda = select_best_lambda_given_C(
            lambda_grid=lambda_grid,
            fba_layer=fba_layer,
            C_init=C_random,
            experimental_fluxes_tensor=median_tensor,
            measured_flux_indices=objective_fluxes_indices,
            num_epochs=10
        )

        print(f"Best lambda found: {best_lambda:.0e}")

        # === Step 5.2: Now do multi-start with best lambda ===
        best_C, _, _ = multi_start_inverse_fba_partial_parallel(
            fba_layer=fba_layer,
            experimental_fluxes_tensor=median_tensor,
            measured_flux_indices=objective_fluxes_indices,
            output_dir=output_dir,
            num_starts=5,
            mode=mode,
            lambda_reg=best_lambda,
            num_epochs=50
        )

        # Plot and save final results
        predicted_fluxes, = fba_layer(best_C)
        predicted_subset = predicted_fluxes[objective_fluxes_indices]
        true_subset = median_tensor[objective_fluxes_indices]
        best_output_dir = os.path.join(output_dir, "final")
        os.makedirs(best_output_dir, exist_ok=True)

        plot_summary(best_C, predicted_subset, true_subset, mode, best_output_dir, objective_fluxes)
        save_flux_and_coefficients(best_C, predicted_subset, true_subset, mode, best_output_dir)



