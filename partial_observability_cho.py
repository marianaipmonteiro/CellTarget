from __future__ import annotations
import argparse
import logging
import os
from typing import List, Tuple, Callable, Sequence, Dict, Any, Set
import cvxpylayers
import pandas as pd
import cobra
from cobra import Model
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
import torch




def plot_grad_per_dimension(grad_dir_history: torch.Tensor, mode: str, labels, output_dir: str, logy: bool = True):
    """
    grad_dir_history: [T, m] tensor of per-epoch gradient norms per measured flux direction.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)
    G = grad_dir_history.cpu().numpy()
    T, m = G.shape

    # plt.figure(figsize=(7,4))
    # for j in range(m):
    #     name = labels[j] if labels is not None and j < len(labels) else f"dir {j}"
    #     plt.plot(G[:, j], label=name, linewidth=1.5)
    # if logy:
    #     plt.yscale("log")
    # plt.xlabel("Epoch")
    # plt.ylabel("|| gradient contribution ||")
    # plt.title(f"Per-dimension gradient magnitude ({mode})")
    # plt.legend(ncol=2, fontsize=8)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f"grad_per_dimension_{mode}.png"), dpi=200)
    # plt.close()

    # print loss per dimension to a csv
    with open(os.path.join(output_dir, f"grad_per_dimension_{mode}.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch"] + [f"dir_{j}" for j in range(m)])
        for t in range(T):
            writer.writerow([t] + list(G[t, :]))


def plot_loss_history(loss_history: list, mode: str,  output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # plt.figure(figsize=(7, 5))
    # plt.plot(loss_history, label='Loss')
    # plt.yscale('log')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss (log scale)')
    # plt.title(f'Loss History ({mode})')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f"loss_history_{mode}.pdf"), format='pdf')
    # plt.close()

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


def _filter_finite_measured_indices(
    experimental_fluxes_tensor: torch.Tensor,  # shape: (n,)
    measured_flux_indices: Sequence[int],
) -> List[int]:
    idx_all: torch.Tensor = torch.tensor(list(measured_flux_indices), device=experimental_fluxes_tensor.device)
    y_all: torch.Tensor = experimental_fluxes_tensor[idx_all]
    keep: torch.Tensor = torch.isfinite(y_all)
    return [int(i) for i in idx_all[keep].tolist()]


def original_optimize_C_partial_lbfgs(
    C: torch.Tensor,                                          # (in) shape: [k], dtype: float32/float64
    fba_layer: Callable[[torch.Tensor], Tuple[torch.Tensor]], # (in) maps C -> (predicted_fluxes,)
    experimental_fluxes_tensor: torch.Tensor,                 # (in) shape: [n], dtype: float32/float64
    measured_flux_indices: Sequence[int],                     # (in) indices into flux vector
    lr: float,                                                # (in) LBFGS step size
    num_epochs: int,                                          # (in) outer iterations
    lambda_reg: float,                                        # (in) L2 weight on C (applied to clamped C)
) -> Tuple[List[float], torch.Tensor, Dict[str, Any]]:
    """
    LBFGS with *soft clamping* C in (0,1) via sigmoid reparameterization.

    Inputs/outputs match the original function signature:
      - Input: initial C (used only to initialize an internal unconstrained theta)
      - Output: best_C returned in (0,1), plus loss_history and extras
    """

    device: torch.device = C.device
    dtype: torch.dtype = C.dtype

    # ---- helpers ----
    def _assert_finite(x: torch.Tensor, name: str) -> None:
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"{name} contains non-finite values.")

    def _theta_to_C(theta: torch.Tensor) -> torch.Tensor:
        eps: float = 1e-6
        theta_safe: torch.Tensor = torch.nan_to_num(theta, nan=0.0, posinf=20.0, neginf=-20.0)
        C01: torch.Tensor = torch.sigmoid(theta_safe)
        return torch.clamp(C01, eps, 1.0 - eps)

    def _safe_forward(C01: torch.Tensor) -> torch.Tensor:
        _assert_finite(C01, "C")
        (pred,) = fba_layer(C01)
        _assert_finite(pred, "predicted_fluxes")
        return pred

    # ---- initialize theta from input C (logit) ----
    eps0: float = 1e-6
    C0: torch.Tensor = torch.clamp(C.detach().clone(), eps0, 1.0 - eps0)
    theta0: torch.Tensor = torch.log(C0 / (1.0 - C0))  # logit

    theta: torch.nn.Parameter = torch.nn.Parameter(theta0, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [theta],
        lr=lr,
        max_iter=10,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    idx: List[int] = _filter_finite_measured_indices(experimental_fluxes_tensor, measured_flux_indices)
    flux_subset: torch.Tensor = experimental_fluxes_tensor[idx]
    m: int = len(idx)

    if m == 0:
        raise ValueError("All measured targets are NaN/Inf after filtering; cannot optimize.")

    loss_history: List[float] = []
    grad_dir_history: List[List[float]] = []  # [T, m]

    @torch.no_grad()
    def eval_loss_only() -> float:
        C01: torch.Tensor = _theta_to_C(theta)
        pred: torch.Tensor = _safe_forward(C01)
        pred_sub: torch.Tensor = pred[idx]
        loss_mse: torch.Tensor = torch.nn.functional.mse_loss(pred_sub, flux_subset)
        loss: torch.Tensor = loss_mse + lambda_reg * torch.sum(C01 ** 2)
        return float(loss.detach().cpu().item())

    best_loss: float = eval_loss_only()
    best_C: torch.Tensor = _theta_to_C(theta).detach().clone()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        try:
            C01: torch.Tensor = _theta_to_C(theta)
            pred: torch.Tensor = _safe_forward(C01)
            pred_sub: torch.Tensor = pred[idx]
            loss: torch.Tensor = torch.nn.functional.mse_loss(pred_sub, flux_subset)
            loss = loss + lambda_reg * torch.sum(C01 ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss is non-finite")
            loss.backward()
            return loss
        except Exception:
            # Always-finite penalty in theta-space (so LBFGS can step away even if theta is corrupted)
            theta_safe: torch.Tensor = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
            penalty: torch.Tensor = 1.0 + 1e-6 * torch.sum(theta_safe ** 2)
            penalty.backward()
            return penalty

    m: int = len(idx)

    for _ in range(num_epochs):
        # ---- per-dimension gradient contributions in C-space (diagnostic) ----
        optimizer.zero_grad(set_to_none=True)
        C01_diag: torch.Tensor = _theta_to_C(theta)

        try:
            pred_diag: torch.Tensor = _safe_forward(C01_diag)
            pred_sub_diag: torch.Tensor = pred_diag[idx]
            residual: torch.Tensor = pred_sub_diag - flux_subset  # [m]

            per_dim_norms: List[float] = []
            for i in range(m):
                unit: torch.Tensor = torch.zeros_like(pred_sub_diag)
                unit[i] = 1.0

                # dpred_i/dC (C-space sensitivity at the current clamped C)
                Ji: torch.Tensor = torch.autograd.grad(
                    outputs=pred_sub_diag,
                    inputs=C01_diag,
                    grad_outputs=unit,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]  # [k]

                gi: torch.Tensor = (2.0 / float(m)) * residual[i] * Ji
                per_dim_norms.append(float(gi.norm().detach().cpu().item()))

            grad_dir_history.append(per_dim_norms)
        except Exception:
            # If the layer fails here, still keep array shapes consistent.
            grad_dir_history.append([float("nan")] * m)
        # ---------------------------------------------------------------------

        optimizer.step(closure)

        post_step_loss: float = eval_loss_only()
        loss_history.append(post_step_loss)

        if post_step_loss < best_loss:
            best_loss = post_step_loss
            best_C = _theta_to_C(theta).detach().clone()

        # Report gradient norm in theta-space (the variable being optimized)
        grad_norm: float = (
            float(theta.grad.norm().detach().cpu().item()) if theta.grad is not None else float("nan")
        )
        print(f"Loss: {loss_history[-1]:.6e} | ∇θ: {grad_norm:.3e}")

    extras: Dict[str, Any] = {
        "grad_dir_history": torch.tensor(grad_dir_history, device="cpu"),
        "best_loss": best_loss,
    }
    return loss_history, best_C, extras


def optimize_C_partial_lbfgs(
    C: torch.Tensor,  # (in) shape: [k], dtype: float32/float64
    fba_layer: Callable[[torch.Tensor], Tuple[torch.Tensor]],  # (in) maps C -> (predicted_fluxes,)
    experimental_fluxes_tensor: torch.Tensor,  # (in) shape: [n], dtype: float32/float64
    measured_flux_indices: Sequence[int],  # (in) indices into flux vector
    lr: float,  # (in) LBFGS step size
    num_epochs: int,  # (in) outer iterations
    lambda_reg: float,  # (in) L2 weight on C (applied to simplex C)
) -> Tuple[List[float], torch.Tensor, Dict[str, Any]]:
    """
    LBFGS with simplex constraint: C = softmax(theta), so sum(C)=1 and C_i>0.

    """

    device: torch.device = C.device
    dtype: torch.dtype = C.dtype

    # ------------------------ helpers ------------------------
    def _assert_finite(x: torch.Tensor, name: str) -> None:
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"{name} contains non-finite values.")

    def _filter_finite_measured_indices(
        y: torch.Tensor, indices: Sequence[int]
    ) -> List[int]:
        idx_out: List[int] = []
        for j in indices:
            if 0 <= int(j) < int(y.numel()):
                v: torch.Tensor = y[int(j)]
                if torch.isfinite(v):
                    idx_out.append(int(j))
        return idx_out

    def _theta_to_C(theta: torch.Tensor) -> torch.Tensor:
        theta_safe: torch.Tensor = torch.nan_to_num(
            theta, nan=0.0, posinf=20.0, neginf=-20.0
        )
        C_simplex: torch.Tensor = torch.softmax(theta_safe, dim=0)  # sum=1
        return C_simplex

    def _safe_forward(C_simplex: torch.Tensor) -> torch.Tensor:
        _assert_finite(C_simplex, "C")
        (pred,) = fba_layer(C_simplex)
        _assert_finite(pred, "predicted_fluxes")
        return pred

    # ------------------- initialize theta --------------------
    eps0: float = 1e-12
    C0: torch.Tensor = torch.clamp(C.detach().clone(), min=eps0)
    if float(C0.sum().detach().cpu().item()) <= 0.0:
        # fallback to uniform init if user passes all-zeros
        k: int = int(C0.numel())
        C0 = torch.full((k,), 1.0 / float(k), device=device, dtype=dtype)
    else:
        C0 = C0 / C0.sum()

    # softmax(log(C0)) == C0 (up to additive constant), good init
    theta0: torch.Tensor = torch.log(C0)
    theta: torch.nn.Parameter = torch.nn.Parameter(theta0, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [theta],
        lr=lr,
        max_iter=10,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    # --------------------- targets subset --------------------
    idx: List[int] = _filter_finite_measured_indices(
        experimental_fluxes_tensor, measured_flux_indices
    )
    flux_subset: torch.Tensor = experimental_fluxes_tensor[idx]
    m: int = int(len(idx))
    if m == 0:
        raise ValueError("All measured targets are NaN/Inf after filtering; cannot optimize.")

    loss_history: List[float] = []
    grad_dir_history: List[List[float]] = []  # [T, m]

    @torch.no_grad()
    def eval_loss_only() -> float:
        C_simplex: torch.Tensor = _theta_to_C(theta)
        pred: torch.Tensor = _safe_forward(C_simplex)
        pred_sub: torch.Tensor = pred[idx]
        loss_mse: torch.Tensor = torch.nn.functional.mse_loss(pred_sub, flux_subset)
        loss: torch.Tensor = loss_mse + lambda_reg * torch.sum(C_simplex ** 2)
        return float(loss.detach().cpu().item())

    best_loss: float = eval_loss_only()
    best_C: torch.Tensor = _theta_to_C(theta).detach().clone()

    # ------------------------ closure ------------------------
    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        try:
            C_simplex: torch.Tensor = _theta_to_C(theta)
            pred: torch.Tensor = _safe_forward(C_simplex)
            pred_sub: torch.Tensor = pred[idx]
            loss_mse: torch.Tensor = torch.nn.functional.mse_loss(pred_sub, flux_subset)
            loss: torch.Tensor = loss_mse + lambda_reg * torch.sum(C_simplex ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss is non-finite")
            loss.backward()
            return loss
        except Exception:
            # Always-finite penalty in theta-space so LBFGS can step away
            theta_safe: torch.Tensor = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
            penalty: torch.Tensor = 1.0 + 1e-6 * torch.sum(theta_safe ** 2)
            penalty.backward()
            return penalty

    # ------------------------ optimize -----------------------
    for _ in range(num_epochs):
        # ---- per-dimension gradient contributions in C-space (diagnostic) ----
        optimizer.zero_grad(set_to_none=True)
        C_diag: torch.Tensor = _theta_to_C(theta)

        try:
            pred_diag: torch.Tensor = _safe_forward(C_diag)
            pred_sub_diag: torch.Tensor = pred_diag[idx]
            residual: torch.Tensor = pred_sub_diag - flux_subset  # [m]

            per_dim_norms: List[float] = []
            for i in range(m):
                unit: torch.Tensor = torch.zeros_like(pred_sub_diag)
                unit[i] = 1.0

                Ji: torch.Tensor = torch.autograd.grad(
                    outputs=pred_sub_diag,
                    inputs=C_diag,
                    grad_outputs=unit,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]  # [k]

                gi: torch.Tensor = (2.0 / float(m)) * residual[i] * Ji
                per_dim_norms.append(float(gi.norm().detach().cpu().item()))

            grad_dir_history.append(per_dim_norms)
        except Exception:
            grad_dir_history.append([float("nan")] * m)
        # ---------------------------------------------------------------------

        optimizer.step(closure)

        post_step_loss: float = eval_loss_only()
        loss_history.append(post_step_loss)

        if post_step_loss < best_loss:
            best_loss = post_step_loss
            best_C = _theta_to_C(theta).detach().clone()

        grad_norm: float = (
            float(theta.grad.norm().detach().cpu().item()) if theta.grad is not None else float("nan")
        )
        print(f"Loss: {loss_history[-1]:.6e} | ∇θ: {grad_norm:.3e}")

    extras: Dict[str, Any] = {
        "grad_dir_history": torch.tensor(grad_dir_history, device="cpu"),
        "best_loss": best_loss,
    }
    return loss_history, best_C, extras



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
    def f(C_tensor: torch.Tensor) -> torch.Tensor:
        try:
            C_tensor = C_tensor.detach().to(dtype=torch.float32, device="cpu").clone().requires_grad_(True)
            loss_history, _, _ = optimize_C_partial_lbfgs(
                C_tensor, fba_layer, experimental_fluxes_tensor, measured_flux_indices,
                lr=1.0, num_epochs=10, lambda_reg=lambda_reg
            )
            y: float = float(loss_history[-1])
            if not np.isfinite(y):
                y = 1e30
            return torch.tensor([y], dtype=torch.float32)
        except Exception:
            return torch.tensor([1e30], dtype=torch.float32)

    dim: int = len(measured_flux_indices)
    lower_bound: torch.Tensor = torch.zeros(dim, dtype=torch.float32, device="cpu")
    upper_bound: torch.Tensor = torch.ones(dim, dtype=torch.float32, device="cpu")

    best_candidate = bayesian_optimization(
        f=f,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_initial_points=5,
        num_iterations=10
    )

    # === Final training ===
    C = best_candidate.clone().detach().requires_grad_(True)

    # # randomly start C between 0 and 1
    # dim = len(measured_flux_indices)
    # C = torch.rand(dim, dtype=torch.float32, requires_grad=True)

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

    measured_labels = [f"flux_{i}" for i in range(len(measured_flux_indices))]  # or real names

    # Compute predictions once
    predicted_fluxes, = fba_layer(final_C.detach())
    predicted_subset = predicted_fluxes[measured_flux_indices]
    experimental_subset = experimental_fluxes_tensor[measured_flux_indices]

    # Save plots/files per run
    plot_grad_per_dimension(extras["grad_dir_history"], mode, [f"flux_{i}" for i in range(len(measured_flux_indices))], run_dir, logy=True)
    #plot_summary(final_C.detach(), predicted_subset, experimental_subset, mode, run_dir, measured_labels)
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

    epsilon: float = 1e-6


    # 1. Biological Score (approx magnitude: 1.0 - 10.0)
    objective_fitness = cp.multiply(C_param, v_measured)
    total_fitness = cp.sum(objective_fitness) + 1e-6  # Avoid div/0

    # 2. Smoothing Penalty (approx magnitude: 0.01 - 0.1)
    smoothing_term = epsilon * cp.sum_squares(v)

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
        objective_expr = -1.0 / total_fitness - smoothing_term
    elif mode == "entropy":
        objective_expr = -cp.sum(cp.entr(objective_fitness + epsilon))  # concave → OK
    else:
        raise ValueError(f"Unknown mode: {mode}")

    #flux_penalty = cp.norm2(v_measured)
    #penalty_weight = 0 #1e-2
    #objective_expr -= penalty_weight * flux_penalty

    v_min_np = v_min.detach().numpy()
    v_max_np = v_max.detach().numpy()
    eps = 0.01  # NOTE: Relaxation of steady-state condition seems to help

    constraints = [
        #S @ v == 0,
        S @ v <= eps,
        -S @ v <= eps,
        v >= v_min_np,
        v <= v_max_np
    ]

    if mode in ["norm", "sum_squares", "logsumexp"]:
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    else: #inverse_log linear entropy
        problem = cp.Problem(cp.Maximize(objective_expr), constraints)

    base_layer = CvxpyLayer(problem, parameters=[C_param], variables=[v])

    return base_layer



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

    # --- LB/UB: keep defaults ---
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
    parser.add_argument("--fluxes_mode", type=str,  help="Fluxes in objective function.")
    args = parser.parse_args()


    file_path = 'inputs/C13_data.xlsx'
    c13 = pd.read_excel(file_path, sheet_name="c13")
    condition = args.condition #'early' # stat, AWZ11expo, AWZ11stat
    arg_output_dir = args.output_dir

    MODEL_PATH: str = "inputs/CHOmpact_cobrapy_original.json"
    NON_PRODUCER_CONDITIONS: Set[str] = {"AWZ11expo", "AWZ11stat"}
    PRODUCT_REACTIONS: List[str] = ["F143"]

    STANDARD_OBJECTIVES: Dict[str, List[str]] = {
        "AWZ11expo": ["F142", "F105", "F107", "F112", "F114"],
        "AWZ11stat": ["F142", "F105", "F107", "F112", "F114"],
        "early": ["F142", "F143", "F105", "F107", "F112", "F114"],
        "stat": ["F142", "F143", "F105", "F107", "F112", "F114"],
    }

    EXTRACELLULAR_OBJ: Dict[str, List[str]] = {
        "AWZ11expo": ['F110','F111','F112','F113','F142','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
        "AWZ11stat": ['F110','F111','F112','F113','F142','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
        "early": ['F110','F143','F111','F112','F113','F142','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
        "stat": ['F110','F143','F111','F112','F113','F142','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
    }

    MEASURED_OBJ: Dict[str, List[str]] = {
        "AWZ11expo": ['F2','F3','F4','F5','F6','F1','F25','F27','F7','F8','F13','F14','F15','F9','F11','F10','F17','F16','F32','F52','F51','F34','F48','F54','F38','F40','F40','F23','F26','F44','F45','F49','F41','F39','F31','F57','F55','F110','F111','F112','F113','F142','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
        "AWZ11stat": ['F2','F3','F4','F5','F6','F1','F25','F27','F7','F8','F13','F14','F15','F9','F11','F10','F17','F16','F32','F52','F51','F34','F48','F54','F38','F40','F40','F23','F26','F44','F45','F49','F41','F39','F31','F57','F55','F110','F111','F112','F113','F142','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
        "early": ['F2','F3','F4','F5','F6','F1','F25','F27','F7','F8','F13','F14','F15','F9','F11','F10','F17','F16','F32','F52','F51','F34','F48','F54','F38','F40','F40','F23','F26','F44','F45','F49','F41','F39','F31','F57','F55','F110','F111','F112','F113','F142','F143','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
        "stat": ['F2','F3','F4','F5','F6','F1','F25','F27','F7','F8','F13','F14','F15','F9','F11','F10','F17','F16','F32','F52','F51','F34','F48','F54','F38','F40','F40','F23','F26','F44','F45','F49','F41','F39','F31','F57','F55','F110','F111','F112','F113','F142','F143','F105','F114','F115','F116','F117','F118','F107','F119','F120','F121','F122','F123','F124','F125','F126','F127','F128'],
    }


    model: Model = load_model(MODEL_PATH)

    if args.fluxes_mode == "all":
        objective_fluxes: List[str] = [f"F{i}" for i in range(1, 145)]
    elif args.fluxes_mode == "standard":
        objective_fluxes = STANDARD_OBJECTIVES[condition]
    elif args.fluxes_mode == "extracellular":
        objective_fluxes = EXTRACELLULAR_OBJ[condition]
    elif args.fluxes_mode == "measured":
        objective_fluxes = MEASURED_OBJ[condition]
    else:
        raise ValueError(f"Unknown fluxes mode: {args.fluxes_mode}")

    lb_row, ub_row, median_row, median_measured_mask = extract_bounds(c13, model, condition)
    flux_to_index = {rxn.id: i for i, rxn in enumerate(model.reactions)}

    measured_flux_indices: list[int] = [
        flux_to_index[rxn_id]
        for rxn_id, is_measured in median_measured_mask.items()
        if bool(is_measured)
    ]

    model, lb_row, ub_row = enforce_non_producer_zero_flux(model,lb_row,ub_row,condition,NON_PRODUCER_CONDITIONS,PRODUCT_REACTIONS)
    constrained_model = constrain_model(model, lb_row, ub_row, median_row)
    median_tensor = torch.tensor(median_row.values, dtype=torch.float32)
    output_dir_input = os.path.join(arg_output_dir,condition)

    num_reactions = len(model.reactions)

    S, vmin, vmax, S_torch, v_min_torch, v_max_torch = initialize_S_and_vs(model, lb_row, ub_row)

    objective_fluxes_indices = [flux_to_index[name] for name in objective_fluxes]
    lambda_grid = [1e-6, 1e-5, 1e-4]

    for mode in ["inverse_log"]: #, "linear", "norm", "sum_squares", "logsumexp"
        output_dir = os.path.join(output_dir_input, mode)
        os.makedirs(output_dir, exist_ok=True)

        fba_layer = setup_optimization_with_basis_functions_partial(
            objective_fluxes_indices, S, v_min_torch, v_max_torch, num_reactions, mode)
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

        save_flux_and_coefficients(best_C, predicted_subset, true_subset, mode, best_output_dir)
        # if args.fluxes_mode == "all":
        #     print("No plotting")
        # else:
        #     plot_summary(best_C, predicted_subset, true_subset, mode, best_output_dir, objective_fluxes)




