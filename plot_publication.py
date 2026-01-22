from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd
import cobra
from cobra import Model
from matplotlib import pyplot as plt

from results_config import OBJ_MAPS, flux_dict, EXTRACELLULAR_MAP_CHO_TO_GEM
import matplotlib as mpl
from plot_chompact_mapping import get_chompact_reaction_dict



# -------------------------
# Types
# -------------------------

mpl.rcParams.update({
    "font.size": 18,              # base
    "axes.titlesize": 28,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
})

Key = Tuple[str, str, str]  # (model_type, data_type, condition)
FluxMeta = Dict[str, Dict[str, Union[int, str]]]


@dataclass(frozen=True)
class RunConfig:
    npj_root: str
    model_paths: Dict[str, str]              # {"cho": "...json", "gem": "...json"}
    model_types: List[str]                   # ["cho"] or ["cho","gem"]
    data_types: List[str]                    # ["standard","extra","measured","all"]
    conditions: List[str]                    # ["AWZ11expo","AWZ11stat","early","stat"]
    mode: str                                # e.g. "inverse_log"
    objfind_pattern: str = "*objfind*.csv"   # pattern for benchmark csv discovery


@dataclass(frozen=True)
class Case:
    """One CellTarget result: one flux_summary.txt."""
    model_type: str
    data_type: str
    condition: str
    mode: str
    coeff_labels: List[str]
    coefficients: List[float]
    v_measured: List[float]
    v_pred: List[float]


@dataclass(frozen=True)
class BenchmarkCase:
    """One ObjFind row from a benchmark CSV."""
    model_type: str
    data_type: str
    condition: str
    source_file: str

    coeff_labels: List[str]
    coefficients: List[float]

    v_measured_labels: List[str]
    v_measured: List[float]
    v_pred_labels: List[str]
    v_pred: List[float]

    loss: float
    status: Any


@dataclass(frozen=True)
class FitMetrics:
    n: int
    mape_percent: float
    smape_percent: float
    r2: float
    bias: float


# -------------------------
# Helpers (pure parsing)
# -------------------------

def flux_tick_label(
    rxn_id: str,                 # type: str
    flux_dict: FluxMeta,         # type: FluxMeta
    include_sign: bool = False,  # type: bool
) -> str:
    meta: Optional[Dict[str, Union[int, str]]] = flux_dict.get(rxn_id)
    if meta is None:
        return rxn_id

    name: str = str(meta.get("reaction", rxn_id))
    if not include_sign:
        return name

    sign_val: int = int(meta.get("sign", 0))  # -1, 0, +1 (as you defined)
    sign_str: str = "−" if sign_val < 0 else ("+" if sign_val > 0 else "0")
    return f"{name} ({sign_str})"


def map_tick_labels(
    rxn_ids: List[str],          # type: List[str]
    flux_dict: FluxMeta,         # type: FluxMeta
    include_sign: bool = False,  # type: bool
) -> List[str]:
    return [flux_tick_label(r, flux_dict, include_sign=include_sign) for r in rxn_ids]


def infer_model_and_data_type(path: Path) -> Tuple[Optional[str], Optional[str]]:
    parts: List[str] = [p.lower() for p in path.parts]
    try:
        npj_idx: int = parts.index("npj")
    except ValueError:
        return None, None

    tail: List[str] = parts[npj_idx + 1 :]

    model_type: Optional[str] = next((p for p in tail if p in {"cho", "gem"}), None)
    data_type: Optional[str] = next((p for p in tail if p in {"standard", "extra", "meas", "measured", "all"}), None)

    # canonicalize "meas" -> "measured"
    if data_type == "meas":
        data_type = "measured"

    return model_type, data_type


def load_flux_summary(filepath: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Returns (v_measured, v_pred, coefficients) from a CellTarget flux_summary.txt.
    """
    def read_block(lines: List[str], start_idx: int) -> Tuple[List[float], int]:
        vals: List[float] = []
        i: int = start_idx
        n: int = len(lines)
        while i < n:
            s: str = lines[i].strip()
            if s == "" or s.endswith(":") or s.startswith("#"):
                break
            vals.extend(float(x) for x in s.split())
            i += 1
        return vals, i

    v_meas: List[float] = []
    v_pred: List[float] = []
    coeffs: List[float] = []

    with open(filepath, "r") as f:
        lines: List[str] = f.readlines()

    i: int = 0
    while i < len(lines):
        line: str = lines[i].strip()

        if line.startswith("Experimental fluxes"):
            v_meas, i = read_block(lines, i + 1)
            continue
        if line.startswith("Predicted fluxes"):
            v_pred, i = read_block(lines, i + 1)
            continue
        if line.startswith("Optimized coefficients"):
            coeffs, i = read_block(lines, i + 1)
            continue

        i += 1

    return v_meas, v_pred, coeffs


# -------------------------
# Objective-label resolution
# -------------------------
def get_obj_labels_for_case(
    model_type: str,
    data_type: str,
    condition: str,
    model: Model,
    obj_maps: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
) -> List[str]:
    """
    Returns reaction IDs used as objective coefficients for this run.
    - For CHO/all: uses model.reactions ordering (robust to F-index assumptions).
    - Otherwise: uses provided OBJ_MAPS lookup.
    """
    if data_type == "all":
        if model_type == "cho":
            return [rxn.id for rxn in model.reactions]
        raise ValueError("data_type='all' is not supported for model_type='gem' in this pipeline.")

    return list(obj_maps[model_type][data_type][condition])


# -------------------------
# CellTarget loaders
# -------------------------

def load_cobra_model(model_path: str) -> cobra.Model:
    """
    Load a COBRA model from JSON (.json) or SBML (.xml / .sbml).

    Parameters
    ----------
    model_path : str

    Returns
    -------
    cobra.Model
    """
    p: Path = Path(model_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suf: str = p.suffix.lower()

    if suf == ".json":
        return cobra.io.load_json_model(str(p))

    if suf in {".xml", ".sbml"}:
        return cobra.io.read_sbml_model(str(p))

    raise ValueError(f"Unsupported model format for: {model_path} (suffix={suf})")

def collect_celltarget_cases(
    cfg: RunConfig,
    obj_maps: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    solver: str = "glpk",
) -> List[Case]:
    cobra.Configuration().solver = solver

    root: Path = Path(cfg.npj_root)
    cases: List[Case] = []

    model_cache: Dict[str, Model] = {}

    for model_type in cfg.model_types:
        if model_type not in model_cache:
            model_cache[model_type] = load_cobra_model(cfg.model_paths[model_type])
        model: Model = model_cache[model_type]

        for data_type in cfg.data_types:
            for condition in cfg.conditions:
                summary_path: Path = root / model_type / data_type / condition / cfg.mode / "final" / "flux_summary.txt"
                if not summary_path.is_file():
                    continue

                v_meas, v_pred, coeffs = load_flux_summary(str(summary_path))
                coeff_labels: List[str] = get_obj_labels_for_case(
                    model_type=model_type,
                    data_type=data_type,
                    condition=condition,
                    model=model,
                    obj_maps=obj_maps,
                )

                cases.append(
                    Case(
                        model_type=model_type,
                        data_type=data_type,
                        condition=condition,
                        mode=cfg.mode,
                        coeff_labels=coeff_labels,
                        coefficients=coeffs,
                        v_measured=v_meas,
                        v_pred=v_pred,
                    )
                )

    return cases


def build_celltarget_index(cases: Sequence[Case]) -> Dict[Key, Case]:
    idx: Dict[Key, Case] = {}
    for c in cases:
        idx[(c.model_type, c.data_type, c.condition)] = c
    return idx


# -------------------------
# ObjFind loaders
# -------------------------
def load_objfind_benchmarks(
    cfg: RunConfig,
    obj_maps: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    benchmark_types: Sequence[str],
    solver: str = "glpk",
) -> List[BenchmarkCase]:
    """
    Reads all benchmark CSVs matching cfg.objfind_pattern under cfg.npj_root.
    Expects columns:
      - condition
      - v_exp_<rxn>, v_pred_<rxn> (multiple)
      - optionally: loss, feasible_status/status
      - optionally: c_<rxn> for coefficients
    """
    cobra.Configuration().solver = solver

    root: Path = Path(cfg.npj_root)
    csv_files: List[Path] = sorted(root.rglob(cfg.objfind_pattern))

    model_cache: Dict[str, Model] = {}
    out: List[BenchmarkCase] = []

    bench_set: set[str] = {str(x) for x in benchmark_types}

    for csv_path in csv_files:
        model_type, data_type = infer_model_and_data_type(csv_path)
        if model_type is None or data_type is None:
            continue
        if model_type not in cfg.model_types:
            continue
        if data_type not in bench_set:
            continue

        if model_type not in model_cache:
            model_cache[model_type] = cobra.io.load_json_model(cfg.model_paths[model_type])
        model: Model = model_cache[model_type]

        df: pd.DataFrame = pd.read_csv(csv_path)

        if "condition" not in df.columns:
            raise ValueError(f"Missing 'condition' column in {csv_path}")

        exp_cols: List[str] = sorted([c for c in df.columns if c.startswith("v_exp_")])
        pred_cols: List[str] = sorted([c for c in df.columns if c.startswith("v_pred_")])
        if len(exp_cols) == 0 or len(pred_cols) == 0:
            raise ValueError(f"Missing v_exp_*/v_pred_* columns in {csv_path}")

        exp_labels: List[str] = [c.replace("v_exp_", "") for c in exp_cols]
        pred_labels: List[str] = [c.replace("v_pred_", "") for c in pred_cols]

        for _, row in df.iterrows():
            condition: str = str(row["condition"])
            if condition not in cfg.conditions:
                continue

            coeff_labels: List[str] = get_obj_labels_for_case(
                model_type=model_type,
                data_type=data_type,
                condition=condition,
                model=model,
                obj_maps=obj_maps,
            )

            coefficients: List[float] = [float(row.get(f"c_{rxn_id}", 0.0)) for rxn_id in coeff_labels]
            v_measured: List[float] = [float(row[c]) for c in exp_cols]
            v_pred: List[float] = [float(row[c]) for c in pred_cols]

            loss: float = float(row["loss"]) if "loss" in df.columns else float("nan")
            status: Any = row.get("feasible_status", row.get("status", None))

            out.append(
                BenchmarkCase(
                    model_type=model_type,
                    data_type=data_type,
                    condition=condition,
                    source_file=str(csv_path),
                    coeff_labels=coeff_labels,
                    coefficients=coefficients,
                    v_measured_labels=exp_labels,
                    v_measured=v_measured,
                    v_pred_labels=pred_labels,
                    v_pred=v_pred,
                    loss=loss,
                    status=status,
                )
            )

    return out


def build_objfind_index(bench_cases: Sequence[BenchmarkCase]) -> Dict[Key, BenchmarkCase]:
    """
    If multiple entries exist for a key, keep the minimum-loss one (if finite).
    """
    idx: Dict[Key, BenchmarkCase] = {}
    for b in bench_cases:
        key: Key = (b.model_type, b.data_type, b.condition)
        if key not in idx:
            idx[key] = b
            continue
        b_loss: float = float(b.loss)
        cur_loss: float = float(idx[key].loss)
        if np.isfinite(b_loss) and (not np.isfinite(cur_loss) or b_loss < cur_loss):
            idx[key] = b
    return idx



def load_data_layer(
    cfg: RunConfig,
    obj_maps: Mapping[str, Mapping[str, Mapping[str, List[str]]]],
    benchmark_types: Sequence[str] = ("standard", "extra"),
    solver: str = "glpk",
) -> Tuple[Dict[Key, Case], Dict[Key, BenchmarkCase]]:
    """
    Returns:
      - ct_idx: CellTarget index keyed by (model_type, data_type, condition)
      - of_idx: ObjFind index keyed by (model_type, data_type, condition)
    """
    ct_cases: List[Case] = collect_celltarget_cases(cfg, obj_maps=obj_maps, solver=solver)
    ct_idx: Dict[Key, Case] = build_celltarget_index(ct_cases)

    of_cases: List[BenchmarkCase] = load_objfind_benchmarks(
        cfg, obj_maps=obj_maps, benchmark_types=list(benchmark_types), solver=solver
    )
    of_idx: Dict[Key, BenchmarkCase] = build_objfind_index(of_cases)

    return ct_idx, of_idx


# -----------------------------
# Core numeric utilities
# -----------------------------
def align_by_labels(
    values: Sequence[float],         # len n_values
    labels: Sequence[str],           # len n_labels
    target_labels: Sequence[str],    # len n_targets
) -> np.ndarray:                     # shape [n_targets]
    out: np.ndarray = np.full(len(target_labels), np.nan, dtype=float)

    n: int = int(min(len(values), len(labels)))
    pos: Dict[str, int] = {str(lab): i for i, lab in enumerate(labels[:n])}

    for i, lab in enumerate(target_labels):
        j: Optional[int] = pos.get(str(lab))
        if j is not None:
            out[i] = float(values[j])
    return out


def compute_fit_metrics(
    y_true: np.ndarray,  # shape [n]
    y_pred: np.ndarray,  # shape [n]
    eps: float = 1e-12,  # for denominators
) -> FitMetrics:
    mask: np.ndarray = np.isfinite(y_true) & np.isfinite(y_pred)
    yt: np.ndarray = y_true[mask].astype(float)
    yp: np.ndarray = y_pred[mask].astype(float)

    n: int = int(yt.size)
    if n == 0:
        return FitMetrics(n=0, mape_percent=np.nan, smape_percent=np.nan, r2=np.nan, bias=np.nan)

    err: np.ndarray = yp - yt
    bias: float = float(np.mean(err))

    denom_mape: np.ndarray = np.maximum(np.abs(yt), eps)
    mape_percent: float = float(100.0 * np.mean(np.abs(err) / denom_mape))

    denom_smape: np.ndarray = np.maximum(np.abs(yt) + np.abs(yp), eps)
    smape_percent: float = float(200.0 * np.mean(np.abs(err) / denom_smape))

    ss_res: float = float(np.sum(err**2))
    ss_tot: float = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2: float = float(1.0 - ss_res / ss_tot) if ss_tot > eps else np.nan

    return FitMetrics(n=n, mape_percent=mape_percent, smape_percent=smape_percent, r2=r2, bias=bias)


def _measured_vector_for_benchmark(
    ct: Case,
    of: BenchmarkCase,
    target_labels: Sequence[str],
) -> np.ndarray:
    """
    Build a single "measured" vector aligned to target_labels.

    Preference:
      1) CellTarget measured (aligned to ct.coeff_labels, then to target_labels)
      2) ObjFind measured (aligned from of.v_measured_labels)
    """
    ct_meas: np.ndarray = align_by_labels(ct.v_measured, ct.coeff_labels, target_labels)
    of_meas: np.ndarray = align_by_labels(of.v_measured, of.v_measured_labels, target_labels)
    measured: np.ndarray = np.where(np.isfinite(ct_meas), ct_meas, of_meas)
    return measured


# -----------------------------
# Section A: table + plot
# -----------------------------

def plot_fluxes_overlay_one_condition(
    ct: "Case",
    of: "BenchmarkCase",
    flux_dict: Mapping[str, Mapping[str, int | str]],
    ax: plt.Axes,
    max_labels: int = 40,
    title: str | None = None,
) -> None:
    """
    Reaction-wise overlay:
      x = flux/reaction label
      y = flux value
      markers = measured / ObjFind / CellTarget
      vertical connectors = |pred - measured| for each method
    """
    target_labels: List[str] = list(ct.coeff_labels)
    if len(target_labels) > int(max_labels):
        target_labels = target_labels[: int(max_labels)]

    ct_meas: np.ndarray = align_by_labels(ct.v_measured, ct.coeff_labels, target_labels)
    ct_pred: np.ndarray = align_by_labels(ct.v_pred, ct.coeff_labels, target_labels)
    of_meas: np.ndarray = align_by_labels(of.v_measured, of.v_measured_labels, target_labels)
    of_pred: np.ndarray = align_by_labels(of.v_pred, of.v_pred_labels, target_labels)

    measured: np.ndarray = np.where(np.isfinite(ct_meas), ct_meas, of_meas)

    x: np.ndarray = np.arange(len(target_labels), dtype=float)

    # points
    ax.scatter(x, measured, s=30, marker="o", label="Measured")
    ax.scatter(x, of_pred, s=30, marker="x", label="ObjFind")
    ax.scatter(x, ct_pred, s=30, marker="^", label="CellTarget")

    of_color: str = "tab:orange"
    ct_color: str = "tab:green"

    for i in range(len(target_labels)):
        if np.isfinite(measured[i]) and np.isfinite(of_pred[i]):
            ax.plot(
                [x[i], x[i]],
                [measured[i], of_pred[i]],
                color=of_color,
                linestyle=":",
                alpha=0.5,
                linewidth=1.2,
            )
        if np.isfinite(measured[i]) and np.isfinite(ct_pred[i]):
            ax.plot(
                [x[i], x[i]],
                [measured[i], ct_pred[i]],
                color=ct_color,
                linestyle="--",
                alpha=0.5,
                linewidth=1.2,
            )


    tick_labels: List[str] = map_tick_labels(target_labels, flux_dict, include_sign=False)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Flux (mmol/[g hr])")

    if title is None:
        title = str(ct.condition)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right")


def plot_coefficients_overlay_one_condition(
    ct: "Case",
    of: "BenchmarkCase",
    flux_dict: Mapping[str, Mapping[str, int | str]],
    ax: plt.Axes,
    top_k: int = 25,
    title: str | None = None,
) -> None:
    """
    Coefficient overlay:
      x = reaction label (objective components)
      y = coefficient value
      bars = ObjFind vs CellTarget

    For large objectives, selects top_k by |CT - OF|.
    """
    # Canonical axis: CellTarget ordering
    target_labels_all: List[str] = list(ct.coeff_labels)

    ct_vals_all: np.ndarray = align_by_labels(ct.coefficients, ct.coeff_labels, target_labels_all)
    of_vals_all: np.ndarray = align_by_labels(of.coefficients, of.coeff_labels, target_labels_all)

    # Select subset to plot (top_k by absolute delta), but keep all if already small
    finite_mask: np.ndarray = np.isfinite(ct_vals_all) & np.isfinite(of_vals_all)
    delta: np.ndarray = np.full_like(ct_vals_all, np.nan, dtype=float)
    delta[finite_mask] = ct_vals_all[finite_mask] - of_vals_all[finite_mask]

    n_all: int = int(len(target_labels_all))
    if n_all > int(top_k):
        # Rank by |delta|, ignore NaNs
        score: np.ndarray = np.where(np.isfinite(delta), np.abs(delta), -np.inf)
        order: np.ndarray = np.argsort(-score)[: int(top_k)]
        order = order[np.isfinite(score[order])]  # drop -inf if any
    else:
        order = np.arange(n_all, dtype=int)

    labels: List[str] = [target_labels_all[int(i)] for i in order.tolist()]
    ct_vals: np.ndarray = ct_vals_all[order]
    of_vals: np.ndarray = of_vals_all[order]

    x: np.ndarray = np.arange(len(labels), dtype=float)
    w: float = 0.38

    ax.bar(x - w / 2, of_vals, width=w, color="tab:orange", alpha=0.85, label="ObjFind")
    ax.bar(x + w / 2, ct_vals, width=w, color="tab:green", alpha=0.85, label="CellTarget")

    tick_labels: List[str] = map_tick_labels(labels, flux_dict, include_sign=False)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")

    ax.set_ylabel("Coefficient value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right")

    if title is None:
        title = str(ct.condition)
    ax.set_title(title)


def report_sectionA_coeff_overlays_2x2(
    ct_idx: Mapping[Key, "Case"],
    of_idx: Mapping[Key, "BenchmarkCase"],
    model_type: str,                     # "cho"
    objective_spaces: Sequence[str],     # ["standard","extra"]
    out_dir: str,
    flux_dict: Mapping[str, Mapping[str, int | str]],
    top_k: int = 25,
) -> List[str]:
    """
    Section A:
    Coefficient overlays in a 2x2 layout (phenotype x phase), one PDF per objective space.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Fixed phenotype × phase layout
    layout: List[List[str]] = [
        ["AWZ11expo", "AWZ11stat"],  # non-producer
        ["early", "stat"],           # producer
    ]

    saved: List[str] = []

    for data_type in objective_spaces:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=False)

        for i in range(2):
            for j in range(2):
                condition: str = layout[i][j]
                ax: plt.Axes = axes[i, j]
                key: Key = (str(model_type), str(data_type), str(condition))

                if key not in ct_idx or key not in of_idx:
                    ax.set_axis_off()
                    ax.set_title(f"{condition} — missing")
                    continue

                # Use smaller top_k for standard automatically
                tk: int = int(top_k)
                if str(data_type) == "standard":
                    tk = 10_000  # plot all (already small)

                plot_coefficients_overlay_one_condition(
                    ct=ct_idx[key],
                    of=of_idx[key],
                    flux_dict=flux_dict,
                    ax=ax,
                    top_k=tk,
                    title=str(condition),
                )

        # Add biological row cues (without messing with axis)
        axes[0, 0].set_ylabel("Non-producer\nCoefficient value")
        axes[1, 0].set_ylabel("Producer\nCoefficient value")

        # fig.suptitle(
        #     f"Coefficients (CHOmpact, {data_type})",
        #     y=1.02,
        # )


        outpath: str = os.path.join(out_dir, f"sectionA_coeff_overlay_2x2_{model_type}_{data_type}.pdf")
        fig.subplots_adjust(
            left=0.14, right=0.98,
            top=0.90, bottom=0.28,
            hspace=0.35, wspace=0.35,
        )
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(outpath)

    return saved


def report_sectionA_flux_overlays(
    ct_idx: Mapping[Key, "Case"],
    of_idx: Mapping[Key, "BenchmarkCase"],
    model_type: str,                     # "cho"
    objective_spaces: Sequence[str],     # ["standard","extra"]
    conditions: Sequence[str],           # ["AWZ11expo","AWZ11stat","early","stat"]
    out_dir: str,
    flux_dict: Mapping[str, Mapping[str, int | str]],
    max_labels: int = 40,
) -> List[str]:
    """
    Section A:
    Reaction-wise flux overlays in a 2x2 layout (phenotype x phase),
    one figure per objective space.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Fixed biological layout
    layout: list[list[str]] = [
        ["AWZ11expo", "AWZ11stat"],  # non-producer
        ["early", "stat"],           # producer
    ]

    for data_type in objective_spaces:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=False)

        for i in range(2):
            for j in range(2):
                condition: str = layout[i][j]
                ax: plt.Axes = axes[i, j]
                key = (model_type, data_type, condition)

                if key not in ct_idx or key not in of_idx:
                    ax.set_axis_off()
                    ax.set_title(f"{condition} — missing")
                    continue

                plot_fluxes_overlay_one_condition(
                    ct=ct_idx[key],
                    of=of_idx[key],
                    flux_dict=flux_dict,
                    ax=ax,
                    max_labels=max_labels,
                    title=condition,
                )

        for ax in axes[:, 1]:
            ax.set_ylabel("")
            ax.yaxis.label.set_visible(False)


        # Row / column labels for biological meaning
        axes[0, 0].set_ylabel("Non-producer\nFlux (mmol/[g hr])")
        axes[1, 0].set_ylabel("Producer\nFlux (mmol/[g hr])")
        axes[1, 0].set_xlabel("Reaction")
        axes[1, 1].set_xlabel("Reaction")

        # fig.suptitle(
        #     f"Flux comparison (CHOmpact, {data_type})",
        #     y=1.02,
        # )

        outpath = os.path.join(
            out_dir, f"sectionA_flux_overlay_2x2_{model_type}_{data_type}.pdf"
        )
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)



# -----------------------------
# Section B
# -----------------------------

def report_sectionB_dimensionality_one_condition_extracellular(
    ct_idx: Mapping[Key, "Case"],
    model_type: str,                     # "cho"
    condition: str,                      # "early"
    objective_spaces: Sequence[str],     # e.g. ["standard","extra","measured","all"]
    out_dir: str,
    flux_dict: Mapping[str, Mapping[str, int | str]],
    x_axis_space: str = "extra",         # defines the extracellular axis labels
    y_symlog: bool = False,
    linthresh: float = 1e-3,
) -> str:
    """
    Section B (CHOmpact only): fixed condition (e.g., 'early') on the extracellular axis.

    Produces one figure:
      columns = objective dimensionality
      row 1   = measured vs predicted extracellular fluxes
      row 2   = inferred coefficients on the same extracellular reactions

    Returns
    -------
    str
        Saved PDF path.
    """
    os.makedirs(out_dir, exist_ok=True)

    axis_key: Key = (str(model_type), str(x_axis_space), str(condition))
    if axis_key not in ct_idx:
        raise KeyError(
            f"Extracellular axis not found: {axis_key}. "
            f"(Need CellTarget run for {model_type}/{x_axis_space}/{condition})"
        )

    # Fix the extracellular reaction axis (reaction IDs)
    axis_labels: List[str] = list(ct_idx[axis_key].coeff_labels)

    n_cols: int = int(len(objective_spaces))
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(6.2 * n_cols, 7.2),  # wider columns + more vertical space
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.88,
        bottom=0.18,
        wspace=0.25,  # horizontal spacing between columns
        hspace=0.25,  # vertical spacing between rows
    )

    for j, data_type in enumerate(objective_spaces):
        key: Key = (str(model_type), str(data_type), str(condition))
        ax_flux: plt.Axes = axes[0, j]
        ax_c: plt.Axes = axes[1, j]

        if key not in ct_idx:
            ax_flux.set_axis_off()
            ax_c.set_axis_off()
            ax_flux.set_title(f"{data_type} — missing")
            continue

        ct: "Case" = ct_idx[key]

        # --- Align measured & predicted fluxes onto extracellular axis
        v_meas: np.ndarray = align_by_labels(ct.v_measured, ct.coeff_labels, axis_labels)
        v_pred: np.ndarray = align_by_labels(ct.v_pred, ct.coeff_labels, axis_labels)

        x: np.ndarray = np.arange(len(axis_labels), dtype=float)

        # (row 1) flux overlay: measured vs predicted
        ax_flux.scatter(x, v_meas, marker="o", label="Measured", alpha=0.9)
        ax_flux.scatter(x, v_pred, marker="^", label="CellTarget", alpha=0.9)

        # optional vertical connectors (constant colour)
        for i in range(len(axis_labels)):
            if np.isfinite(v_meas[i]) and np.isfinite(v_pred[i]):
                ax_flux.plot(
                    [x[i], x[i]],
                    [v_meas[i], v_pred[i]],
                    linestyle="--",
                    alpha=0.35,
                    linewidth=1.0,
                )

        if y_symlog:
            ax_flux.set_yscale("symlog", linthresh=linthresh)

        ax_flux.set_title(str(data_type))
        ax_flux.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax_flux.set_ylabel("Flux (mmol/[g hr])" if j == 0 else "")

        # (row 2) coefficients on the SAME axis
        c_vals: np.ndarray = align_by_labels(ct.coefficients, ct.coeff_labels, axis_labels)
        ax_c.bar(x, c_vals, color="tab:orange", width=0.7, alpha=0.85)
        ax_c.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax_c.set_ylabel("Coefficient value" if j == 0 else "")

        # x tick labels only on bottom row
        tick_labels: List[str] = map_tick_labels(axis_labels, flux_dict, include_sign=False)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            fontsize=12,
        )
        ax_flux.set_xticks([])

    # single legend (top-left panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    fig.suptitle(
        f"Effect of objective dimensionality (CHOmpact, {condition}, extracellular axis)",
        y=1.03,
    )

    outpath: str = os.path.join(out_dir, f"sectionB_dimensionality_{model_type}_{condition}_extracellular.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


# -----------------------------
# Section C
# -----------------------------

def plot_sectionC_standard_and_extra_one_fig(
    ct_idx: Mapping["Key", "Case"],
    condition: str,
    out_dir: str,
    extracellular_map: Mapping[str, Tuple[str, str]],
    flux_dict_cho: Mapping[str, Mapping[str, int | str]] | None = None,
    y_symlog_flux: bool = False,
    linthresh: float = 1e-3,
) -> str:
    """
    Section C: Effect of model resolution (CHOmpact vs iCHO2441),
    shown for both objective sets (standard, extra).
    """

    os.makedirs(out_dir, exist_ok=True)

    data_types: List[str] = ["standard", "extra"]

    for dt in data_types:
        if ("cho", dt, condition) not in ct_idx:
            raise KeyError(f"Missing case: ('cho', {dt}, {condition})")
        if ("gem", dt, condition) not in ct_idx:
            raise KeyError(f"Missing case: ('gem', {dt}, {condition})")

    labels: List[str] = list(extracellular_map.keys())
    cho_ids: List[str] = [extracellular_map[k][0] for k in labels]
    gem_ids: List[str] = [extracellular_map[k][1] for k in labels]

    # --- horizontal spacing between metabolites
    dx: float = 1.5
    x: np.ndarray = dx * np.arange(len(labels), dtype=float)

    # --- colors (semantic, consistent)
    c_meas: str = "black"
    c_cho: str = "#1f77b4"   # blue
    c_gem: str = "#ff7f0e"   # orange

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.5, 8.5),   # larger figure → more x-tick room
        sharey=False,
        constrained_layout=False,
    )

    legend_handles: list[Any] = []

    for j, dt in enumerate(data_types):
        ct_cho: "Case" = ct_idx[("cho", dt, condition)]
        ct_gem: "Case" = ct_idx[("gem", dt, condition)]

        # =========================
        # Top row: fluxes
        # =========================
        ax_flux = axes[0, j]

        cho_meas = align_by_labels(ct_cho.v_measured, ct_cho.coeff_labels, cho_ids)
        cho_pred = align_by_labels(ct_cho.v_pred, ct_cho.coeff_labels, cho_ids)
        gem_meas = align_by_labels(ct_gem.v_measured, ct_gem.coeff_labels, gem_ids)
        gem_pred = align_by_labels(ct_gem.v_pred, ct_gem.coeff_labels, gem_ids)

        meas = np.where(np.isfinite(cho_meas), cho_meas, gem_meas)

        h_meas = ax_flux.scatter(
            x, meas, color=c_meas, marker="o", alpha=0.9,
            label=("Measured" if j == 0 else "_nolegend_"),
        )
        h_cho = ax_flux.scatter(
            x, cho_pred, color=c_cho, marker="^", alpha=0.9,
            label=("CellTarget (CHOmpact)" if j == 0 else "_nolegend_"),
        )
        h_gem = ax_flux.scatter(
            x, gem_pred, color=c_gem, marker="s", alpha=0.9,
            label=("CellTarget (iCHO2441)" if j == 0 else "_nolegend_"),
        )

        for i in range(len(x)):
            if np.isfinite(meas[i]) and np.isfinite(cho_pred[i]):
                ax_flux.plot([x[i], x[i]], [meas[i], cho_pred[i]],
                             linestyle="--", color=c_cho, alpha=0.35)
            if np.isfinite(meas[i]) and np.isfinite(gem_pred[i]):
                ax_flux.plot([x[i], x[i]], [meas[i], gem_pred[i]],
                             linestyle=":", color=c_gem, alpha=0.35)

        if y_symlog_flux:
            ax_flux.set_yscale("symlog", linthresh=linthresh)

        ax_flux.set_title(dt)
        ax_flux.set_xticks(x)
        ax_flux.set_xticklabels(labels, rotation=45, ha="right")
        ax_flux.grid(True, axis="y", linestyle="--", alpha=0.3)
        if j == 0:
            ax_flux.set_ylabel("Flux (mmol/[g hr])")
        if j == 1:  # extra
            ymax: float = 1.1 * float(np.nanmax(np.r_[cho_c, gem_c]))
            ax_c.set_ylim(0.0, ymax)

        # =========================
        # Bottom row: coefficients
        # =========================
        ax_c = axes[1, j]

        cho_c = align_by_labels(ct_cho.coefficients, ct_cho.coeff_labels, cho_ids)
        gem_c = align_by_labels(ct_gem.coefficients, ct_gem.coeff_labels, gem_ids)

        w: float = 0.38 * dx
        bars_cho = ax_c.bar(
            x - w / 2, cho_c, width=w, color=c_cho, alpha=0.9,
            label=("CHOmpact (coeff.)" if j == 0 else "_nolegend_"),
        )
        bars_gem = ax_c.bar(
            x + w / 2, gem_c, width=w, color=c_gem, alpha=0.9,
            label=("iCHO2441 (coeff.)" if j == 0 else "_nolegend_"),
        )

        ax_c.set_xticks(x)
        ax_c.set_xticklabels(labels, rotation=45, ha="right")
        ax_c.grid(True, axis="y", linestyle="--", alpha=0.3)
        if j == 0:
            ax_c.set_ylabel("Objective coefficient")

        if j == 0:
            legend_handles = [h_meas, h_cho, h_gem, bars_cho, bars_gem]

    #fig.suptitle("Effect of Model Resolution", y=0.97)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        wspace=0.32,
        hspace=0.45,
        bottom=0.26,
        top=0.90,
    )

    outpath: str = os.path.join(
        out_dir, f"sectionC_model_resolution_{condition}_standard_extra.pdf"
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return outpath


# -----------------------------
# Section D
# -----------------------------


# -----------------------------
# Section D — NEW helper
# -----------------------------
def build_sectionD_model_cache(
    model_paths: Mapping[str, str],   # (in) e.g. {"cho": "...json", "gem": "...xml"}
    solver: str = "glpk",             # (in)
) -> Dict[str, Model]:
    """
    Load the COBRA models needed for Section D metadata (reaction name/equation).

    Returns
    -------
    Dict[str, cobra.Model]
        keys: model_type ("cho","gem")
    """
    cobra.Configuration().solver = str(solver)
    cache: Dict[str, Model] = {}
    for model_type, path in model_paths.items():
        cache[str(model_type)] = load_cobra_model(str(path))  # uses your existing loader
    return cache


def _safe_rxn_meta_from_model(
    model: Model,     # (in)
    rxn_id: str,      # (in)
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (rxn_name, rxn_equation) if rxn_id exists in model; otherwise (None, None).
    """
    try:
        rxn = model.reactions.get_by_id(str(rxn_id))
        return str(rxn.name), str(rxn.reaction)
    except Exception:
        return None, None


# -----------------------------
# Section D — unchanged helper
# -----------------------------
def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# -----------------------------
# Section D — MODIFIED
# -----------------------------
def build_sectionD_coefficients_long_df(
    ct_idx: Mapping[Tuple[str, str, str], "Case"],              # (in)
    flux_dict: Mapping[str, Mapping[str, int | str]],           # (in)
    model_cache: Mapping[str, Model],                           # (in)
    chompact_name_map: Mapping[str, str] | None = None,         # (in)
    include_abs_rank: bool = True,                              # (in)
) -> pd.DataFrame:
    """
    Long-form coefficient table.

    Policy:
      - CHOmpact  → rxn_name = (chompact_name_map[rxn_id] if provided else rxn_id)
      - iCHO2441 → rxn_name = COBRA name (fallback: flux_dict, then ID)
    """
    rows: List[Dict[str, Any]] = []

    for (model_type, data_type, condition), ct in ct_idx.items():
        model: Optional[Model] = model_cache.get(str(model_type))

        n: int = int(min(len(ct.coeff_labels), len(ct.coefficients)))
        coeffs: np.ndarray = np.array(
            [_safe_float(ct.coefficients[i]) for i in range(n)],
            dtype=float,
        )

        if include_abs_rank and n > 0:
            order: np.ndarray = np.argsort(-np.abs(coeffs))
            abs_rank: np.ndarray = np.empty(n, dtype=int)
            abs_rank[order] = np.arange(1, n + 1, dtype=int)
        else:
            abs_rank = None

        for i in range(n):
            try:
                rxn_id: str = str(ct.coeff_labels[i])
                meta: Mapping[str, int | str] = flux_dict.get(rxn_id, {})
                rxn_sign: int = int(meta.get("sign", 0))

                rxn_equation: str = ""

                if str(model_type).lower() == "cho":
                    # CHOmpact: use mapping if available, else keep F###
                    rxn_name: str = (
                        str(chompact_name_map.get(rxn_id, rxn_id))
                        if chompact_name_map is not None
                        else rxn_id
                    )
                else:
                    # iCHO2441 (or others): prefer COBRA name, fallback to flux_dict
                    rxn_name = rxn_id
                    if model is not None:
                        try:
                            rxn = model.reactions.get_by_id(rxn_id)
                            if rxn.name and str(rxn.name).strip():
                                rxn_name = str(rxn.name)
                            if rxn.reaction:
                                rxn_equation = str(rxn.reaction)
                        except KeyError:
                            pass

                    if rxn_name == rxn_id:
                        rxn_name = str(meta.get("reaction", rxn_id))

                row: Dict[str, Any] = {
                    "model_type": str(model_type),
                    "data_type": str(data_type),
                    "condition": str(condition),
                    "rxn_id": rxn_id,
                    "rxn_name": rxn_name,
                    "rxn_equation": rxn_equation,
                    "rxn_sign": rxn_sign,
                    "coefficient": float(coeffs[i]),
                }

                if include_abs_rank:
                    row["abs_coefficient"] = float(np.abs(coeffs[i]))
                    row["abs_rank_within_case"] = (
                        int(abs_rank[i]) if abs_rank is not None else np.nan
                    )

                rows.append(row)

            except Exception as e:
                # Keep the export robust: skip problematic rows, do not crash the whole table.
                print(
                    f"[SectionD] Skip: ({model_type},{data_type},{condition}) "
                    f"i={i} rxn={ct.coeff_labels[i]} err={e}"
                )
                continue

    df: pd.DataFrame = pd.DataFrame(rows)

    # If collection failed, return an empty but well-formed table (avoids KeyError on sort)
    expected_cols: List[str] = [
        "model_type",
        "data_type",
        "condition",
        "rxn_id",
        "rxn_name",
        "rxn_equation",
        "rxn_sign",
        "coefficient",
    ]
    if include_abs_rank:
        expected_cols += ["abs_coefficient", "abs_rank_within_case"]

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    sort_cols: List[str] = ["model_type", "data_type", "condition"]
    sort_cols += (
        ["abs_rank_within_case"]
        if include_abs_rank and "abs_rank_within_case" in df.columns
        else ["rxn_id"]
    )

    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

# -----------------------------
# Section D — OPTIONAL MODIFICATION (recommended)
# -----------------------------
def build_sectionD_coefficients_wide_df(
    coeff_long_df: pd.DataFrame,                                            # (in)
    case_cols: Sequence[str] = ("model_type", "data_type", "condition"),     # (in)
    rxn_cols: Sequence[str] = ("rxn_id", "rxn_name", "rxn_equation", "rxn_sign"),  # (in) MODIFIED default
    values_col: str = "coefficient",                                        # (in)
) -> pd.DataFrame:
    """
    Wide matrix now carries rxn_equation too (so the CSV is self-explanatory).
    """
    df: pd.DataFrame = coeff_long_df.copy()

    df["case_key"] = df[list(case_cols)].astype(str).apply(
        lambda r: "/".join(r.values.tolist()),
        axis=1,
    )

    wide: pd.DataFrame = (
        df.pivot_table(
            index=list(rxn_cols),
            columns="case_key",
            values=values_col,
            aggfunc="first",
        )
        .reset_index()
    )

    fixed: List[str] = list(rxn_cols)
    case_keys: List[str] = sorted([c for c in wide.columns if c not in fixed])
    wide = wide[fixed + case_keys]
    return wide


# -----------------------------
# Section D — MODIFIED (labels)
# -----------------------------

def plot_sectionD_coeff_heatmap_faceted(
    long_df: pd.DataFrame,                 # (in)
    out_dir: str,                          # (in)
    *,
    k: int = 25,                           # (in)
    normalize: str = "l1",                 # (in) {"none","l1"}
    use_abs: bool = True,                  # (in)
) -> None:
    """
    One heatmap per (model_type, condition).

    Rows = top-k reactions by mean |c|
    Columns = objective space (data_type)

    Label policy:
      - CHOmpact  → F###
      - iCHO2441 → FOO: reaction name
    """
    os.makedirs(out_dir, exist_ok=True)

    for (model_type, condition), g in long_df.groupby(
        ["model_type", "condition"], dropna=False
    ):
        # map rxn_id -> rxn_name
        rxn_name_map: Dict[str, str] = (
            g.drop_duplicates(subset=["rxn_id"])[["rxn_id", "rxn_name"]]
            .set_index("rxn_id")["rxn_name"]
            .astype(str)
            .to_dict()
        )

        # rxn × objective-space matrix
        P: pd.DataFrame = g.pivot_table(
            index="rxn_id",
            columns="data_type",
            values="coefficient",
            aggfunc="first",
        )

        if use_abs:
            P = P.abs()

        if normalize == "l1":
            denom: pd.Series = P.sum(axis=0).replace(0.0, np.nan)
            P = P.div(denom, axis=1)

        # select top-k reactions
        score: pd.Series = P.mean(axis=1)
        P = P.loc[score.sort_values(ascending=False).head(int(k)).index]

        # -----------------------
        # y-axis labels (FIXED)
        # -----------------------
        rxn_ids: List[str] = [str(rid) for rid in P.index]

        ylabels: List[str] = []
        for rid in rxn_ids:
            name: str = rxn_name_map.get(rid, rid)

            # collapse CHOmpact duplicates
            if name.strip() == rid.strip():
                ylabels.append(rid)
            else:
                ylabels.append(f"{rid}: {name}")

        # -----------------------
        # plotting
        # -----------------------
        fig, ax = plt.subplots(
            figsize=(7.6, 0.35 * len(P) + 2.5)
        )

        im = ax.imshow(
            P.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
        )

        # IMPORTANT: set ticks AND labels
        ax.set_yticks(np.arange(len(ylabels), dtype=int))
        ax.set_yticklabels(ylabels, fontsize=9)

        ax.set_xticks(np.arange(len(P.columns), dtype=int))
        ax.set_xticklabels(
            [str(c) for c in P.columns],
            rotation=45,
            ha="right",
        )

        title_model: str = (
            "CHOmpact" if str(model_type).lower() == "cho" else str(model_type)
        )
        ax.set_title(f"{title_model} – {str(condition)}")

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                out_dir,
                f"sectionD_coeff_heatmap_{model_type}_{condition}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


# -----------------------------
# Section D — MODIFIED
# -----------------------------

def compute_sectionD_reaction_order(
    long_df: pd.DataFrame,                 # (in)
    *,
    score: str = "mean_abs",               # (in) {"mean_abs","median_abs","freq_nonzero"}
    nonzero_eps: float = 1e-12,            # (in)
) -> pd.DataFrame:
    """
    Compute a global reaction order across *all* cases in long_df.

    Returns
    -------
    pd.DataFrame
        Columns: rxn_id, rxn_name, rxn_sign, score
        Sorted descending by score.
    """
    g = long_df.groupby(["rxn_id", "rxn_name", "rxn_sign"], dropna=False)["coefficient"]

    if score == "mean_abs":
        s = g.apply(lambda x: float(np.nanmean(np.abs(x.to_numpy(dtype=float)))))
    elif score == "median_abs":
        s = g.apply(lambda x: float(np.nanmedian(np.abs(x.to_numpy(dtype=float)))))
    elif score == "freq_nonzero":
        s = g.apply(lambda x: float(np.nanmean(np.abs(x.to_numpy(dtype=float)) > float(nonzero_eps))))
    else:
        raise ValueError(f"Unknown score: {score}")

    out: pd.DataFrame = s.reset_index().rename(columns={"coefficient": "score"})
    out = out.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def apply_reaction_order_to_wide_df(
    wide_df: pd.DataFrame,                 # (in)
    order_df: pd.DataFrame,                # (in) output of compute_sectionD_reaction_order
) -> pd.DataFrame:
    """
    Reorder wide_df rows by the reaction order.

    Returns
    -------
    pd.DataFrame
    """
    order_ids: List[str] = order_df["rxn_id"].astype(str).tolist()
    rank: Dict[str, int] = {rid: i for i, rid in enumerate(order_ids)}

    tmp: pd.DataFrame = wide_df.copy()
    tmp["__ord__"] = tmp["rxn_id"].astype(str).map(rank).fillna(10**12).astype(int)
    tmp = tmp.sort_values("__ord__", kind="mergesort").drop(columns="__ord__").reset_index(drop=True)
    return tmp

def plot_sectionD_horizontal_heatmaps_by_condition(
    long_df: pd.DataFrame,                      # (in)
    out_dir: str,                               # (in)
    *,
    model_type: str,                            # (in) "cho" or "gem"
    conditions: Sequence[str],                  # (in) e.g. ["AWZ11expo","AWZ11stat","early","stat"]
    objective_spaces: Sequence[str],            # (in) CHO: ("standard","extra","measured","all"); GEM: ("standard","extra")
    k: int = 80,                                # (in)
    normalize: str = "none",                      # (in) {"none","l1"}
    use_abs: bool = True,                       # (in)
    eps: float = 0.0,                           # (in) drop rows with max<=eps after normalization
    label_policy: str = "compact",              # (in) {"compact","id_name"}
    drop_all_nan_cols: bool = True,             # (in)
) -> None:
    """
    Creates one *horizontal* heatmap per condition for a fixed model_type.

    Rows    = top-k reactions by mean magnitude across objective spaces.
    Columns = objective spaces (data_type).
    """
    os.makedirs(out_dir, exist_ok=True)

    dfm: pd.DataFrame = long_df[long_df["model_type"].astype(str) == str(model_type)].copy()
    if dfm.empty:
        print(f"[SectionD] No rows for model_type={model_type}")
        return

    # rxn_id -> rxn_name for y-labels
    rxn_name_map: Dict[str, str] = (
        dfm.drop_duplicates(subset=["rxn_id"])[["rxn_id", "rxn_name"]]
        .set_index("rxn_id")["rxn_name"]
        .astype(str)
        .to_dict()
    )

    def _make_ylabels(rxn_ids: List[str]) -> List[str]:
        out: List[str] = []
        for rid in rxn_ids:
            nm: str = rxn_name_map.get(rid, rid)
            if label_policy == "compact":
                if str(model_type).lower() == "cho":
                    out.append(rid)  # CHOmpact: keep F###
                else:
                    out.append(rid if nm.strip() == rid.strip() else f"{rid}: {nm}")
            elif label_policy == "id_name":
                out.append(rid if nm.strip() == rid.strip() else f"{rid}: {nm}")
            else:
                raise ValueError(f"Unknown label_policy: {label_policy}")
        return out

    for cond in conditions:
        dff: pd.DataFrame = dfm[dfm["condition"].astype(str) == str(cond)].copy()
        if dff.empty:
            print(f"[SectionD] No rows for model_type={model_type}, condition={cond}")
            continue

        # rxn × objective-space
        P: pd.DataFrame = dff.pivot_table(
            index="rxn_id",
            columns="data_type",
            values="coefficient",
            aggfunc="first",
        )

        # enforce desired objective-space order
        P = P.reindex(columns=[str(x) for x in objective_spaces], copy=False)

        if drop_all_nan_cols:
            P = P.dropna(axis=1, how="all")

        if use_abs:
            P = P.abs()

        if normalize == "l1":
            denom: pd.Series = P.sum(axis=0).replace(0.0, np.nan)
            P = P.div(denom, axis=1)

        if float(eps) > 0.0 and not P.empty:
            P = P.loc[P.max(axis=1) > float(eps)]

        if P.empty:
            print(f"[SectionD] Empty after filtering: model_type={model_type}, condition={cond}")
            continue

        # top-k reactions
        score: pd.Series = P.mean(axis=1)
        P = P.loc[score.sort_values(ascending=False).head(int(k)).index]

        n_rows: int = int(P.shape[0])
        n_cols: int = int(P.shape[1])

        fig_w: float = float(max(8.0, 1.15 * n_cols + 5.0))
        fig_h: float = float(max(6.0, 0.22 * n_rows + 2.8))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(P.to_numpy(dtype=float), aspect="auto", interpolation="nearest")

        rxn_ids: List[str] = [str(x) for x in P.index]
        ax.set_yticks(np.arange(n_rows, dtype=int))
        ax.set_yticklabels(_make_ylabels(rxn_ids), fontsize=8)

        ax.set_xticks(np.arange(n_cols, dtype=int))
        ax.set_xticklabels([str(c) for c in P.columns], rotation=45, ha="right", fontsize=10)

        title_model: str = "CHOmpact" if str(model_type).lower() == "cho" else "iCHO2441"
        ax.set_title(f"{title_model} – {str(cond)}")

        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        fig.tight_layout()

        outpath: str = os.path.join(out_dir, f"sectionD_horizontal_{model_type}_{cond}.pdf")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)



def export_sectionD_coefficients_tables(
    ct_idx: Mapping[Tuple[str, str, str], "Case"],                 # (in)
    flux_dict: Mapping[str, Mapping[str, int | str]],              # (in)
    out_dir: str,                                                  # (in)
    basename: str = "sectionD_coefficients",                       # (in)
    *,
    model_paths: Mapping[str, str],                                # (in) NEW
    solver: str = "glpk",                                          # (in) NEW
    order_score: str = "mean_abs",                                 # (in)
    make_faceted_heatmaps: bool = True,                            # (in)
    heatmap_k: int = 25,                                           # (in)
) -> Tuple[str, str, str]:
    """
    Now loads models (CHO/GEM) to attach reaction names and equations.
    """
    os.makedirs(out_dir, exist_ok=True)
    CHOMPACT_RXN_NAME: Dict[str, str] = get_chompact_reaction_dict()
    model_cache: Dict[str, Model] = build_sectionD_model_cache(model_paths=model_paths, solver=str(solver))
    long_df = build_sectionD_coefficients_long_df(
        ct_idx=ct_idx,
        flux_dict=flux_dict,
        model_cache=model_cache,
        chompact_name_map=CHOMPACT_RXN_NAME,
        include_abs_rank=True,
    )

    conds = ["AWZ11expo", "AWZ11stat", "early", "stat"]

    # CHOmpact (has all 4 objective spaces)
    plot_sectionD_horizontal_heatmaps_by_condition(
        long_df=long_df,
        out_dir="NPJ_publication/sectionD_coefficients",
        model_type="cho",
        conditions=conds,
        objective_spaces=("standard", "extra", "measured", "all"),
        k=80,
        normalize="none",
        use_abs=True,
        label_policy="compact",
    )

    # iCHO2441 (only standard + extra; measured/all will be auto-dropped anyway)
    plot_sectionD_horizontal_heatmaps_by_condition(
        long_df=long_df,
        out_dir="NPJ_publication/sectionD_coefficients",
        model_type="gem",
        conditions=conds,
        objective_spaces=("standard", "extra"),
        k=80,
        normalize="none",
        use_abs=True,
        label_policy="compact",
        drop_all_nan_cols=True,  # removes measured/all if empty
    )

    order_df: pd.DataFrame = compute_sectionD_reaction_order(long_df, score=str(order_score))

    wide_df: pd.DataFrame = build_sectionD_coefficients_wide_df(long_df)
    wide_df = apply_reaction_order_to_wide_df(wide_df, order_df)

    long_csv: str = os.path.join(out_dir, f"{basename}_long.csv")
    wide_csv: str = os.path.join(out_dir, f"{basename}_wide.csv")
    order_csv: str = os.path.join(out_dir, f"{basename}_reaction_order.csv")

    long_df.to_csv(long_csv, index=False)
    wide_df.to_csv(wide_csv, index=False)
    order_df.to_csv(order_csv, index=False)

    # if bool(make_faceted_heatmaps):
    #     plot_sectionD_coeff_heatmap_faceted(
    #         long_df=long_df,
    #         out_dir=out_dir,
    #         k=int(heatmap_k),
    #         normalize="l1",
    #         use_abs=True
    #     )

    return long_csv, wide_csv, order_csv

def plot_sectionD_horizontal_heatmaps_by_phase(
    long_df: pd.DataFrame,                      # (in)
    out_dir: str,                               # (in)
    *,
    model_type: str,                            # (in) "cho" or "gem"
    phases: Mapping[str, Sequence[str]],        # (in) e.g. {"expo":["AWZ11expo","early"], "stat":["AWZ11stat","stat"]}
    objective_spaces: Sequence[str] = ("standard", "extra", "measured", "all"),  # (in)
    k: int = 60,                                # (in) reactions shown
    normalize: str = "l1",                      # (in) {"none","l1"}
    use_abs: bool = True,                       # (in)
    eps: float = 0.0,                           # (in) drop rows with max<=eps after normalization
    label_policy: str = "compact",              # (in) {"compact","id_name"}
) -> None:
    """
    For a fixed model_type, creates one wide heatmap per phase.
    Columns = (condition × objective space).
    Rows = top-k reactions by mean magnitude across all shown columns.

    label_policy:
      - "compact": CHO uses rxn_id; GEM uses "id: name"
      - "id_name": always "id: name"
    """
    os.makedirs(out_dir, exist_ok=True)

    dfm: pd.DataFrame = long_df[long_df["model_type"].astype(str) == str(model_type)].copy()
    if dfm.empty:
        print(f"[SectionD] No rows for model_type={model_type}")
        return

    # rxn_id -> rxn_name map (for y labels)
    rxn_name_map: Dict[str, str] = (
        dfm.drop_duplicates(subset=["rxn_id"])[["rxn_id", "rxn_name"]]
        .set_index("rxn_id")["rxn_name"]
        .astype(str)
        .to_dict()
    )

    def _make_ylabels(rxn_ids: List[str]) -> List[str]:
        out: List[str] = []
        for rid in rxn_ids:
            nm: str = rxn_name_map.get(rid, rid)
            if label_policy == "compact":
                if str(model_type).lower() == "cho":
                    out.append(rid)
                else:
                    out.append(rid if nm.strip() == rid.strip() else f"{rid}: {nm}")
            elif label_policy == "id_name":
                out.append(rid if nm.strip() == rid.strip() else f"{rid}: {nm}")
            else:
                raise ValueError(f"Unknown label_policy: {label_policy}")
        return out

    for phase_name, conds in phases.items():
        dff: pd.DataFrame = dfm[dfm["condition"].astype(str).isin([str(c) for c in conds])].copy()
        if dff.empty:
            print(f"[SectionD] No rows for model_type={model_type}, phase={phase_name}, conds={conds}")
            continue

        # Build matrix: rows rxn_id; cols (condition, data_type)
        P: pd.DataFrame = dff.pivot_table(
            index="rxn_id",
            columns=["condition", "data_type"],
            values="coefficient",
            aggfunc="first",
        )

        # enforce column order (condition × objective_spaces)
        desired_cols: List[Tuple[str, str]] = []
        for c in conds:
            for dt in objective_spaces:
                desired_cols.append((str(c), str(dt)))
        P = P.reindex(columns=pd.MultiIndex.from_tuples(desired_cols), copy=False)

        if use_abs:
            P = P.abs()

        if normalize == "l1":
            denom: pd.Series = P.sum(axis=0).replace(0.0, np.nan)
            P = P.div(denom, axis=1)

        if float(eps) > 0.0:
            P = P.loc[P.max(axis=1) > float(eps)]

        # score + top-k
        score: pd.Series = P.mean(axis=1)
        P = P.loc[score.sort_values(ascending=False).head(int(k)).index]

        # plot
        n_rows: int = int(P.shape[0])
        n_cols: int = int(P.shape[1])

        # wide figure: width scales with columns; height with rows
        fig_w: float = float(max(10.0, 1.05 * n_cols + 6.0))
        fig_h: float = float(max(6.0, 0.22 * n_rows + 2.8))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(P.to_numpy(dtype=float), aspect="auto", interpolation="nearest")

        rxn_ids: List[str] = [str(x) for x in P.index]
        ax.set_yticks(np.arange(n_rows, dtype=int))
        ax.set_yticklabels(_make_ylabels(rxn_ids), fontsize=8)

        xlabels: List[str] = [f"{c}/{dt}" for (c, dt) in P.columns.to_list()]
        ax.set_xticks(np.arange(n_cols, dtype=int))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)

        title_model: str = "CHOmpact" if str(model_type).lower() == "cho" else "iCHO2441"
        ax.set_title(f"{title_model} – {phase_name}")

        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        fig.tight_layout()

        outpath: str = os.path.join(out_dir, f"sectionD_horizontal_{model_type}_{phase_name}.pdf")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)


# -----------------------------
# Section E — Loss plots
# -----------------------------

def _resolve_loss_file(
    run_dir: str,                       # type: str
    mode: str,                          # type: str
) -> Optional[str]:
    """
    Your structure:
      .../<mode>/run_*/loss_history_<mode>.csv
    """
    p: Path = Path(run_dir) / f"loss_history_{mode}.csv"
    return str(p) if p.is_file() else None



def find_loss_run_dirs(
    base_dir: str,   # type: str
    mode: str,       # type: str
) -> List[str]:
    """
    Finds .../<mode>/run_*/loss_history_<mode>.csv
    """
    p: Path = Path(base_dir)
    if not p.is_dir():
        return []

    out: List[str] = []
    for d in sorted(p.iterdir()):
        if not (d.is_dir() and d.name.startswith("run_")):
            continue
        if (d / f"loss_history_{mode}.csv").is_file():
            out.append(str(d))
    return out


def _read_loss_curve(
    loss_path: str,  # type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts either:
      - columns: Epoch, Loss
      - or a 1-col numeric file (interpreted as Loss, Epoch inferred)
    """
    df: pd.DataFrame = pd.read_csv(loss_path)

    if ("Epoch" in df.columns) and ("Loss" in df.columns):
        x: np.ndarray = df["Epoch"].to_numpy(dtype=float)
        y: np.ndarray = df["Loss"].to_numpy(dtype=float)
        return x, y

    # fallback: first numeric column as Loss
    numeric_cols: List[str] = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 1:
        y = df[numeric_cols[0]].to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        return x, y

    raise ValueError(f"Unrecognized loss file format: {loss_path}")


def plot_loss_overview_for_condition(
    base_dir: str,                 # type: str
    out_dir: str,                  # type: str
    mode: str,                     # type: str
    title: str,                    # type: str
    loss_file: str = "loss.csv",   # type: str
    logy: bool = True,             # type: bool
) -> Optional[str]:
    """
    Produces ONE overview PDF for a condition, overlaying all runs found under base_dir.

    Returns the saved path, or None if no runs exist.
    """
    run_dirs: List[str] = find_loss_run_dirs(base_dir, mode=mode)
    if len(run_dirs) == 0:
        return None

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    legend_labels: List[str] = []

    for i, run in enumerate(run_dirs, start=1):
        loss_path: Optional[str] = _resolve_loss_file(run, mode=mode)
        if loss_path is None:
            continue
        try:
            x, y = _read_loss_curve(loss_path)
            ax.plot(x, y)
            run_name: str = os.path.basename(run.rstrip("/\\")) or f"run_{i}"
            legend_labels.append(f"Run {i}")
        except Exception as e:
            print(f"[WARN] Failed to read loss curve: {loss_path} ({e})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")

    if legend_labels:
        ax.legend(legend_labels, frameon=False)

    out_path: str = os.path.join(out_dir, f"sectionE_loss_overview_{mode}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def report_sectionE_loss_overviews_2x2(
    cfg: RunConfig,
    model_type: str,                     # type: str
    objective_spaces: Sequence[str],     # type: Sequence[str]
    out_dir: str,                        # type: str
    loss_file: str = "loss.csv",         # type: str
    logy: bool = True,                   # type: bool
) -> List[str]:
    """
    Section E:
    For each data_type in objective_spaces, produce a 2x2 figure overlaying losses per condition.
    Layout matches Section A:
      [AWZ11expo  AWZ11stat]
      [early      stat    ]
    """
    os.makedirs(out_dir, exist_ok=True)

    layout: List[List[str]] = [
        ["AWZ11expo", "AWZ11stat"],
        ["early", "stat"],
    ]

    saved: List[str] = []

    for data_type in objective_spaces:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=False)

        for i in range(2):
            for j in range(2):
                condition: str = layout[i][j]
                ax: plt.Axes = axes[i, j]

                base_dir: str = str(Path(cfg.npj_root) / model_type / data_type / condition / cfg.mode)
                run_dirs: List[str] = find_loss_run_dirs(base_dir,  mode=cfg.mode)

                if len(run_dirs) == 0:
                    ax.set_axis_off()
                    ax.set_title(f"{condition} — missing")
                    continue

                legend_labels: List[str] = []
                for k, run in enumerate(run_dirs, start=1):
                    loss_path: Optional[str] = _resolve_loss_file(run,  mode=cfg.mode)
                    if loss_path is None:
                        continue
                    try:
                        x, y = _read_loss_curve(loss_path)
                        ax.plot(x, y)
                        run_name: str = os.path.basename(run.rstrip("/\\")) or f"run_{k}"
                        legend_labels.append(f"Run {k}")
                    except Exception as e:
                        print(f"[WARN] Failed to read loss curve: {loss_path} ({e})")

                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(condition)
                if logy:
                    ax.set_yscale("log")
                if legend_labels:
                    ax.legend(legend_labels, frameon=False, fontsize=14)

                ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        axes[0, 0].set_ylabel("Non-producer\nLoss")
        axes[1, 0].set_ylabel("Producer\nLoss")

        outpath: str = os.path.join(out_dir, f"sectionE_loss_overlay_2x2_{model_type}_{data_type}.pdf")
        fig.subplots_adjust(
            left=0.08, right=0.98,
            top=0.92, bottom=0.14,
            hspace=0.35, wspace=0.35,
        )
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(outpath)

    return saved


def main() -> None:
    p = argparse.ArgumentParser()

    # inputs
    p.add_argument("--npj-root", type=str, default="NPJ")
    p.add_argument("--mode", type=str, default="inverse_log")
    p.add_argument("--objfind-pattern", type=str, default="*objfind*.csv")
    p.add_argument("--solver", type=str, default="glpk")

    # selections
    p.add_argument("--model-types", nargs="+", type=str, default=["cho","gem"])
    p.add_argument("--data-types", nargs="+", type=str, default=["standard", "extra", "measured", "all"])
    p.add_argument("--conditions", nargs="+", type=str, default=["AWZ11expo", "AWZ11stat", "early", "stat"])
    p.add_argument("--benchmark-types", nargs="+", type=str, default=["standard", "extra"])

    a = p.parse_args()

    model_paths: Dict[str, str] = {
        "cho": "inputs/CHOmpact_cobrapy_original.json",
        "gem": "inputs/iCHO2441.xml",
    }

    cfg = RunConfig(
        npj_root=str(a.npj_root),
        model_paths=model_paths,
        model_types=list(a.model_types),
        data_types=list(a.data_types),
        conditions=list(a.conditions),
        mode=str(a.mode),
        objfind_pattern=str(a.objfind_pattern),
    )

    ct_idx, of_idx = load_data_layer(
        cfg=cfg,
        obj_maps=OBJ_MAPS,
        benchmark_types=list(a.benchmark_types),
        solver=str(a.solver),
    )

    # Minimal data-layer sanity output (no analysis/plots)
    print(f"CellTarget cases: {len(ct_idx)}")
    print(f"ObjFind cases:    {len(of_idx)}")
    print(f"Matched keys:     {len(set(ct_idx.keys()) & set(of_idx.keys()))}")

    # Optional: basic integrity check
    bad: int = 0
    for key, ct in ct_idx.items():
        if len(ct.coefficients) != len(ct.coeff_labels):
            bad += 1
    if bad:
        print(f"WARNING: {bad} CellTarget cases have len(coefficients) != len(coeff_labels).")

    # Section A — CHOmpact benchmark vs ObjFind using reaction-wise flux overlays
    _ = report_sectionA_flux_overlays(
        ct_idx=ct_idx,
        of_idx=of_idx,
        model_type="cho",
        objective_spaces=["standard", "extra"],
        conditions=["AWZ11expo", "AWZ11stat", "early", "stat"],
        out_dir="NPJ_publication/sectionA_objfind_C1",
        flux_dict=flux_dict,
        max_labels=40,
    )

    _ = report_sectionA_coeff_overlays_2x2(
        ct_idx=ct_idx,
        of_idx=of_idx,
        model_type="cho",
        objective_spaces=["standard", "extra"],
        out_dir="NPJ_publication/sectionA_objfind_C1",
        flux_dict=flux_dict,
        top_k=25,  # good default for "extra"
    )

    # Section B — Coeeficients across dimensionality
    # _ = report_sectionB_dimensionality_one_condition_extracellular(
    #     ct_idx=ct_idx,
    #     model_type="cho",
    #     condition="early",
    #     objective_spaces=["standard", "extra", "measured", "all"],
    #     out_dir="NPJ_publication/sectionB_dimensionality_C1",
    #     flux_dict=flux_dict,
    #     x_axis_space="extra",
    #     y_symlog=False,
    # )
    #
    # # Section C
    # _ = plot_sectionC_standard_and_extra_one_fig(
    #     ct_idx=ct_idx,
    #     condition="early",
    #     out_dir="NPJ_publication/sectionC_modeltype",
    #     extracellular_map=EXTRACELLULAR_MAP_CHO_TO_GEM,
    #     y_symlog_flux=False,
    # )
    #
    # # Section D
    # long_csv, wide_csv, order_csv = export_sectionD_coefficients_tables(
    #     ct_idx=ct_idx,
    #     flux_dict=flux_dict,
    #     out_dir="NPJ_publication/sectionD_coefficients",
    #     basename="sectionD_coefficients",
    #     model_paths=model_paths,
    #     solver=str(a.solver),  # optional
    #     heatmap_k=25,
    # )

    # Section E — Loss plots (CellTarget)
    # _ = report_sectionE_loss_overviews_2x2(
    #     cfg=cfg,
    #     model_type="cho",
    #     objective_spaces=["standard", "extra"],  # or any subset in cfg.data_types
    #     out_dir="NPJ_publication/sectionE_loss",
    #     logy=True,
    # )





if __name__ == "__main__":
    main()





