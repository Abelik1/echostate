#region Physical TESTS
import numpy as np
from echostate.utils import mean_absolute_error
import torch
def summarize(arr):
    arr = np.asarray(arr)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max())
    }

def check_bounds(z_pred, eps=1e-6):
    z_pred = np.asarray(z_pred)
    violations = np.where(np.abs(z_pred) > 1 + eps)[0]
    max_excess = float((np.abs(z_pred) - 1).clip(min=0).max()) if violations.size else 0.0
    return {
        "num_violations": int(violations.size),
        "violation_indices_sample": violations[:10].tolist(),
        "max_excess_over_1": max_excess
    }

def autocorr_lag1(x):
    x = np.asarray(x)
    x = x - x.mean()
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(np.correlate(x[:-1], x[1:])[0] / ((len(x)-1)*x.std()*x.std()))

def magnetization_from_qubit_series(z_per_qubit, t_len=None):
    """
    z_per_qubit: dict {q: np.array(seq_len)} for same timeline (after washout alignment if needed).
    returns: Mz(t) array
    """
    qs = sorted(z_per_qubit.keys())
    Z = np.stack([z_per_qubit[q][:t_len] for q in qs], axis=0)  # (Q, T)
    return Z.sum(axis=0)

def drift_stats(series):
    """How constant is a series? report mean abs diff and slope via linear fit."""
    y = np.asarray(series)
    diffs = np.abs(np.diff(y))
    # simple slope
    t = np.arange(len(y))
    slope = float(np.polyfit(t, y, 1)[0]) if len(y) >= 2 else 0.0
    return {"mean_abs_step": float(diffs.mean() if len(diffs) else 0.0), "linear_slope": slope}

def compare_series(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    T = min(len(y_pred), len(y_true))
    y_pred = y_pred[:T]; y_true = y_true[:T]
    mae = mean_absolute_error(torch.tensor(y_pred), torch.tensor(y_true)).item()
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    return {"mae": float(mae), "rmse": rmse}

def render_physics_report(json_path="./examples/Heisenberg_Chain/cache/physics_summary.json",
                          out_dir=None,
                          show=True):
    """
    Read physics_summary.json and produce a compact visual report:
      - Per-(N,qubit) ESN MAE/RMSE
      - Bounds violations (|<σz>|>1)
      - Simulator norm & energy drift summaries
      - Purity statistics
      - Global magnetization summaries (if present)

    Creates a multi-page PDF and a few PNGs in out_dir, and prints a concise text summary.
    """
    import os, json, math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(json_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Load ----------
    with open(json_path, "r") as f:
        rows = json.load(f)

    # ---------- Buckets ----------
    # per-qubit ESN diag
    esn_rows = []   # dicts with keys: N, qubit, series_error{mae,rmse}, pred_bounds{num_violations,...}, summaries, acf1...
    # simulator per-qubit diag
    sim_rows = []   # dicts inside {"simulator_checks": {...}}
    # global magnetization rows
    mag_rows = []   # dicts inside {"global_magnetization": {...}}

    for item in rows:
        if "simulator_checks" in item:
            sim_rows.append(item["simulator_checks"])
        elif "global_magnetization" in item:
            mag = item["global_magnetization"]
            # inject N if missing
            if "N" not in mag:
                mag["N"] = None
            mag_rows.append(mag)
        else:
            # assume per-qubit ESN diag shape
            # require minimal keys to count it as ESN diag
            if all(k in item for k in ["N", "qubit", "series_error", "pred_bounds"]):
                esn_rows.append(item)

    # helper to tag
    def tag(n, q):
        return f"N{n}_q{q}"

    # ---------- Aggregate helpers ----------
    def collect_esn_metric(name):
        xs, vals = [], []
        for r in esn_rows:
            xs.append(tag(r["N"], r["qubit"]))
            vals.append(float(r["series_error"].get(name, np.nan)))
        return xs, np.array(vals)

    def collect_bounds():
        xs, cnts, max_exc = [], [], []
        for r in esn_rows:
            xs.append(tag(r["N"], r["qubit"]))
            pb = r.get("pred_bounds", {})
            cnts.append(int(pb.get("num_violations", 0)))
            max_exc.append(float(pb.get("max_excess_over_1", 0.0)))
        return xs, np.array(cnts), np.array(max_exc)

    def collect_sim_drift(key="energy_drift", sub="linear_slope"):
        xs, vals = [], []
        for r in sim_rows:
            xs.append(tag(r["N"], r["qubit"]))
            vals.append(abs(float(r.get(key, {}).get(sub, 0.0))))
        return xs, np.array(vals)

    def collect_purity(summary_stat="mean"):
        xs, vals = [], []
        for r in sim_rows:
            xs.append(tag(r["N"], r["qubit"]))
            vals.append(float(r.get("purity_summary", {}).get(summary_stat, np.nan)))
        return xs, np.array(vals)

    # ---------- Prepare data ----------
    x_mae_labels, mae_vals = collect_esn_metric("mae")
    _, rmse_vals = collect_esn_metric("rmse")
    x_bounds_labels, bound_cnts, bound_maxexc = collect_bounds()
    x_energy_labels, energy_slope = collect_sim_drift("energy_drift", "linear_slope")
    x_norm_labels, norm_step = collect_sim_drift("norm_drift", "mean_abs_step")
    x_purity_mean, purity_mean = collect_purity("mean")
    _, purity_std = collect_purity("std")
    _, purity_min = collect_purity("min")
    _, purity_max = collect_purity("max")

    # ---------- Plotting ----------
    pdf_path = os.path.join(out_dir, "physics_summary_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # 1) ESN error bars
        fig, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(len(x_mae_labels))
        ax.bar(idx - 0.2, mae_vals, width=0.4, label="MAE")
        ax.bar(idx + 0.2, rmse_vals, width=0.4, label="RMSE")
        ax.set_xticks(idx)
        ax.set_xticklabels(x_mae_labels, rotation=45, ha="right")
        ax.set_title("ESN Prediction Error per (N, qubit)")
        ax.set_ylabel("Error")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_esn_errors.png")); plt.close(fig)

        # 2) Bounds violations
        fig, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(len(x_bounds_labels))
        ax.bar(idx, bound_cnts, label="#(|⟨σz⟩|>1)")
        ax.plot(idx, bound_maxexc, marker="o", linestyle="--", label="Max excess over 1")
        ax.set_xticks(idx)
        ax.set_xticklabels(x_bounds_labels, rotation=45, ha="right")
        ax.set_title("ESN Physical Bounds Checks")
        ax.set_ylabel("Count / Excess")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_bounds.png")); plt.close(fig)

        # 3) Simulator drifts (energy & norm)
        fig, ax = plt.subplots(figsize=(10, 4))
        idxE = np.arange(len(x_energy_labels))
        ax.bar(idxE - 0.2, energy_slope, width=0.4, label="|Energy slope|")
        idxN = np.arange(len(x_norm_labels))
        ax.bar(idxN + 0.2, norm_step, width=0.4, label="Mean |Δ Norm|")
        xticks = list(dict.fromkeys(x_energy_labels + x_norm_labels))  # preserve order and uniqueness
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, ha="right")
        ax.set_title("Simulator Unitarity Checks")
        ax.set_ylabel("Drift")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_drift.png")); plt.close(fig)

        # 4) Purity summary (mean ± range)
        fig, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(len(x_purity_mean))
        ax.bar(idx, purity_mean, width=0.5, label="Purity mean")
        # error bars as min/max whiskers
        err_low = purity_mean - purity_min
        err_high = purity_max - purity_mean
        ax.errorbar(idx, purity_mean, yerr=[err_low, err_high], fmt="none", capsize=4, label="Range (min–max)")
        ax.set_xticks(idx)
        ax.set_xticklabels(x_purity_mean, rotation=45, ha="right")
        ax.set_title("Single-Qubit Purity (from simulator ρ_k)")
        ax.set_ylabel("Tr(ρ²)")
        ax.set_ylim(0.45, 1.05)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_purity.png")); plt.close(fig)

        # 5) Global magnetization summaries (if present)
        if mag_rows:
            # One page per N (or once if N is None)
            Ns = sorted(set(m.get("N") for m in mag_rows))
            for n in Ns:
                mags = [m for m in mag_rows if m.get("N") == n]
                # take first (they're summaries)
                m = mags[0]
                def fmt_summary(prefix, d):
                    return f"{prefix} mean={d.get('mean', 'NA'):.4g}, std={d.get('std','NA'):.4g}, slope={m.get(prefix+'_drift',{}).get('linear_slope',0):.3e}"
                fig, ax = plt.subplots(figsize=(8, 3))
                parts = []
                if "mag_true_summary" in m:
                    parts.append(fmt_summary("mag_true", m["mag_true_summary"]))
                if "mag_pred_summary" in m:
                    parts.append(fmt_summary("mag_pred", m["mag_pred_summary"]))
                if "mag_series_error" in m:
                    parts.append(f"MAE={m['mag_series_error'].get('mae',np.nan):.4g}, RMSE={m['mag_series_error'].get('rmse',np.nan):.4g}")
                text = "\n".join(parts) if parts else "No magnetization summary found."
                ax.axis("off")
                ax.set_title(f"Global Magnetization Summary (N={n})")
                ax.text(0.02, 0.5, text, va="center", ha="left", family="monospace")
                fig.tight_layout()
                pdf.savefig(fig); fig.savefig(os.path.join(out_dir, f"mag_summary_N{n}.png")); plt.close(fig)

    # ---------- Console summary ----------
    def topk(labels, values, k=5, reverse=False):
        order = np.argsort(values)
        if reverse:
            order = order[::-1]
        order = order[:k]
        return [(labels[i], float(values[i])) for i in order]

    print("\n=== Physics Report ===")
    if len(mae_vals):
        worst_mae = topk(x_mae_labels, mae_vals, k=min(5, len(mae_vals)), reverse=True)
        print("Worst MAE (top 5):")
        for lbl, val in worst_mae:
            print(f"  {lbl}: {val:.4g}")
    if len(bound_cnts) and bound_cnts.sum() > 0:
        offenders = [(x_bounds_labels[i], int(bound_cnts[i])) for i in np.where(bound_cnts>0)[0]]
        print("Bounds violations (|<σz⟩| > 1):")
        for lbl, cnt in offenders:
            print(f"  {lbl}: {cnt}")
    if len(energy_slope):
        worst_energy = topk(x_energy_labels, energy_slope, k=min(5, len(energy_slope)), reverse=True)
        print("Largest |energy slope| (top 5):")
        for lbl, val in worst_energy:
            print(f"  {lbl}: {val:.3e}")
    if len(norm_step):
        worst_norm = topk(x_norm_labels, norm_step, k=min(5, len(norm_step)), reverse=True)
        print("Largest mean |Δ norm| (top 5):")
        for lbl, val in worst_norm:
            print(f"  {lbl}: {val:.3e}")

    print(f"\nSaved report PDF → {pdf_path}")
    print(f"PNGs saved in     → {out_dir}")

    if show:
        # Open the last PNG (arbitrary) to pop a window in interactive sessions
        try:
            import webbrowser
            webbrowser.open_new(pdf_path)
        except Exception:
            pass

import json
from pathlib import Path
from typing import Union, Dict, Any, List

def scorecard_physics(summary_json: Union[str, Path, Dict, List],
                      *,
                      # ---- thresholds you can tweak ----
                      norm_std_tol=1e-12,
                      norm_slope_tol=1e-15,
                      energy_std_tol=1e-10,
                      energy_slope_tol=1e-12,
                      purity_min_floor=0.0,
                      purity_max_ceiling=1.0,
                      mae_ok=0.03, mae_warn=0.08,
                      rmse_ok=0.04, rmse_warn=0.10,
                      acf_diff_ok=0.01, acf_diff_warn=0.03,
                      mag_slope_tol=5e-6,
                      quiet=False) -> Dict[str, Any]:
    """
    Read a physics_summary.json (either a path or an already-loaded list/dict)
    and print a concise scorecard, returning a structured summary.

    Status semantics:
      ✅ = pass (green)   | ⚠️ = caution (yellow) | ❌ = fail (red)

    Thresholds are defaults you can tune to your system’s scale.
    """
    def _status(ok: bool=None, warn: bool=False):
        if ok is True:  return "✅"
        if warn:        return "⚠️"
        return "❌"

    # Load JSON
    if isinstance(summary_json, (str, Path)):
        data = json.loads(Path(summary_json).read_text())
    else:
        data = summary_json

    # Normalize list of blocks
    if isinstance(data, dict):
        blocks = [data]
    else:
        blocks = list(data)

    # Extract blocks by type
    sim_checks = [b["simulator_checks"] for b in blocks if "simulator_checks" in b]
    mag_block  = next((b["global_magnetization"] for b in blocks if "global_magnetization" in b), None)
    qubit_series = [b for b in blocks if all(k in b for k in ("qubit","pred_summary","true_summary","series_error","acf1_pred","acf1_true","pred_bounds"))]

    report = {"simulator": {}, "qubits": {}, "magnetization": {}, "overall": {"ok": True, "notes": []}}

    # ---------- Simulator checks per qubit ----------
    for sc in sim_checks:
        q = sc["qubit"]
        ns = sc["norm_summary"]
        es = sc["energy_summary"]
        nd = sc["norm_drift"]
        ed = sc["energy_drift"]
        ps = sc.get("purity_summary", {})
        # Norm
        norm_ok  = (ns["std"] <= norm_std_tol) and (abs(nd["linear_slope"]) <= norm_slope_tol)
        norm_warn = (ns["std"] <= 10*norm_std_tol) and (abs(nd["linear_slope"]) <= 10*norm_slope_tol)
        # Energy
        en_ok  = (es["std"] <= energy_std_tol) and (abs(ed["linear_slope"]) <= energy_slope_tol)
        en_warn = (es["std"] <= 10*energy_std_tol) and (abs(ed["linear_slope"]) <= 10*energy_slope_tol)
        # Purity in range
        purity_ok = (ps.get("min", 0.0) >= purity_min_floor - 1e-9) and (ps.get("max", 1.0) <= purity_max_ceiling + 1e-9)

        report["simulator"][q] = {
            "norm":   {"status": _status(ok=norm_ok, warn=norm_warn), "std": ns["std"], "slope": nd["linear_slope"]},
            "energy": {"status": _status(ok=en_ok, warn=en_warn), "std": es["std"], "slope": ed["linear_slope"]},
            "purity": {"status": _status(ok=purity_ok), "mean": ps.get("mean"), "min": ps.get("min"), "max": ps.get("max")},
        }

    # ---------- Per-qubit ESN series ----------
    for qb in qubit_series:
        q = qb["qubit"]
        mae  = qb["series_error"]["mae"]
        rmse = qb["series_error"]["rmse"]
        acf_diff = abs(qb["acf1_pred"] - qb["acf1_true"])
        bounds_ok = (qb["pred_bounds"]["num_violations"] == 0)

        # MAE / RMSE grading
        mae_status  = _status(ok=(mae <= mae_ok),  warn=(mae_ok < mae <= mae_warn))
        rmse_status = _status(ok=(rmse <= rmse_ok), warn=(rmse_ok < rmse <= rmse_warn))
        acf_status  = _status(ok=(acf_diff <= acf_diff_ok), warn=(acf_diff_ok < acf_diff <= acf_diff_warn))
        bounds_status = _status(ok=bounds_ok)

        report["qubits"][q] = {
            "mae":  {"status": mae_status,  "value": mae},
            "rmse": {"status": rmse_status, "value": rmse},
            "acf1_diff": {"status": acf_status, "value": acf_diff},
            "bounds": {"status": bounds_status, "violations": qb["pred_bounds"]["num_violations"]},
            "pred_summary": qb["pred_summary"],
            "true_summary": qb["true_summary"],
        }

    # ---------- Global magnetization ----------
    if mag_block:
        mtrue = mag_block["mag_true_summary"]
        mpred = mag_block["mag_pred_summary"]
        mte   = mag_block["mag_true_drift"]["linear_slope"]
        mpe   = mag_block["mag_pred_drift"]["linear_slope"]
        m_mae = mag_block["mag_series_error"]["mae"]
        m_rmse= mag_block["mag_series_error"]["rmse"]

        drift_ok = (abs(mte) <= mag_slope_tol and abs(mpe) <= mag_slope_tol)
        report["magnetization"] = {
            "drift": {"status": _status(ok=drift_ok, warn=abs(mte)<=10*mag_slope_tol and abs(mpe)<=10*mag_slope_tol),
                      "true_slope": mte, "pred_slope": mpe},
            "error": {
                "mae":  {"status": _status(ok=(m_mae<=mae_ok),  warn=(mae_ok<m_mae<=mae_warn)),  "value": m_mae},
                "rmse": {"status": _status(ok=(m_rmse<=rmse_ok), warn=(rmse_ok<m_rmse<=rmse_warn)), "value": m_rmse},
            },
            "true_summary": mtrue, "pred_summary": mpred,
        }

    # ---------- Print summary ----------
    if not quiet:
        print("\n=== Physics Summary Scorecard ===")
        # Simulator
        print("\n-- Simulator checks (per qubit) --")
        for q in sorted(report["simulator"].keys()):
            s = report["simulator"][q]
            print(f"Qubit {q}: "
                  f"Norm {s['norm']['status']} (std={s['norm']['std']:.2e}, slope={s['norm']['slope']:.2e}) | "
                  f"Energy {s['energy']['status']} (std={s['energy']['std']:.2e}, slope={s['energy']['slope']:.2e}) | "
                  f"Purity {s['purity']['status']} (mean={s['purity']['mean']:.3f}, "
                  f"range=[{s['purity']['min']:.3f},{s['purity']['max']:.3f}])")

        # Qubit series
        print("\n-- ESN predictions vs. truth (per qubit) --")
        for q in sorted(report["qubits"].keys()):
            s = report["qubits"][q]
            print(f"Qubit {q}: "
                  f"MAE {s['mae']['status']}={s['mae']['value']:.4f} | "
                  f"RMSE {s['rmse']['status']}={s['rmse']['value']:.4f} | "
                  f"ACFΔ {s['acf1_diff']['status']}={s['acf1_diff']['value']:.4f} | "
                  f"Bounds {s['bounds']['status']} (violations={s['bounds']['violations']})")

        # Magnetization
        if report["magnetization"]:
            m = report["magnetization"]
            print("\n-- Global magnetization --")
            print(f"Drift {m['drift']['status']} (true slope={m['drift']['true_slope']:.2e}, "
                  f"pred slope={m['drift']['pred_slope']:.2e}) | "
                  f"MAE {m['error']['mae']['status']}={m['error']['mae']['value']:.4f} | "
                  f"RMSE {m['error']['rmse']['status']}={m['error']['rmse']['value']:.4f}")

        print("\nLegend: ✅ pass | ⚠️ caution | ❌ fail\n")

    # Overall ok?
    any_fail = False
    for section in ("simulator","qubits","magnetization"):
        sec = report.get(section, {})
        if isinstance(sec, dict):
            for _, v in sec.items():
                if isinstance(v, dict):
                    # look for nested 'status'
                    for vv in v.values() if "status" not in v else [v]:
                        if isinstance(vv, dict) and "status" in vv and vv["status"] == "❌":
                            any_fail = True
                elif "status" in v and v["status"] == "❌":
                    any_fail = True
    report["overall"]["ok"] = not any_fail
    return report


    # Fallback: infer from attributes that are present
    res = getattr(model, "reservoir", None)
    trainer = getattr(model, "trainer", None)
    cfg = dict(
        device=str(getattr(model, "device", "cpu")),
        base_input_dim=getattr(model, "base_input_dim", 1),
        reservoir_size=getattr(model, "reservoir_size",
                               getattr(res, "reservoir_size", 100)),
        output_dim=getattr(model, "output_dim", 1),
        feedback=getattr(model, "feedback", 0),
        spectral_radius=getattr(model, "spectral_radius",
                                getattr(res, "spectral_radius", 0.9)),
        sparsity=getattr(model, "sparsity",
                         getattr(res, "sparsity", 0.1)),
        input_scaling=getattr(model, "input_scaling", 1.0),
        bias_scaling=getattr(model, "bias_scaling", 0.0),
        ridge_param=getattr(trainer, "ridge_param", 1e-6),
        learning_algo=getattr(trainer, "learning_algo", "inv"),
        leak_rate=getattr(model, "leak_rate", 1.0),
        washout=getattr(model, "washout", 50),
        batch_size=getattr(model, "batch_size", 1),
        seed=getattr(model, "seed", None),
        step_log_every=getattr(model, "step_log_every", None),
        profile=getattr(model, "profile", False),
    )
    return cfg

if __name__ == "__main__":
    from glob import glob
    import re
    import torch
    from echostate.ESN import *
    from .run_esn_sim import *
    def _extract_seeds_from_name(path: str):
        """Parse reservoir seed (rSeedXXX) and global seed (SeedXXX) from filename."""
        fname = path.split("/")[-1].split("\\")[-1]  # works cross-platform
        m_r = re.search(r"rSeed(\d+)", fname)
        m_s = re.search(r"Seed(\d+)", fname)
        return {
            "reservoir_seed": int(m_r.group(1)) if m_r else None,
            "seed": int(m_s.group(1)) if m_s else None,
        }

    def _clone_into_new_esn(old_m, seeds: dict):
        cfg = _safe_get_config(old_m)

        # Inject reservoir seed from filename if present
        if seeds.get("reservoir_seed") is not None:
            cfg["seed"] = seeds["reservoir_seed"]
        elif seeds.get("seed") is not None and cfg.get("seed") is None:
            cfg["seed"] = seeds["seed"]

        new = ESN(**cfg)

        # Copy reservoir matrices
        old_res = getattr(old_m, "reservoir", None)
        new_res = getattr(new, "reservoir", None)
        if old_res is not None and new_res is not None:
            for name in ("W_in", "W_bias", "W"):
                if hasattr(old_res, name) and hasattr(new_res, name):
                    getattr(new_res, name).data.copy_(
                        getattr(old_res, name).detach().to(new_res.device)
                    )

        # --- Copy trained readout (robust) ---
        if getattr(old_m, "W_out", None) is not None:
            W = old_m.W_out.detach().to(new.device)
            existing = getattr(new, "W_out", None)
            if isinstance(existing, torch.Tensor):
                existing.data.copy_(W)
            else:
                if hasattr(new, "W_out"):
                    try:
                        delattr(new, "W_out")
                    except Exception:
                        pass
                try:
                    new.register_buffer("W_out", W)
                except Exception:
                    new.W_out = W

        # Mirror trainer fields
        if hasattr(old_m, "trainer") and hasattr(new, "trainer"):
            if hasattr(old_m.trainer, "ridge_param"):
                new.trainer.ridge_param = float(old_m.trainer.ridge_param)
            if hasattr(old_m.trainer, "learning_algo"):
                new.trainer.learning_algo = str(old_m.trainer.learning_algo)

        return new


    def migrate_legacy_models(glob_pattern, *, overwrite=True):
        from glob import glob
        for p in glob(glob_pattern):
            try:
                old = torch.load(p, map_location="cpu", weights_only=False)
                if not isinstance(old, ESN):
                    print("Skip (not ESN):", p)
                    continue

                seeds = _extract_seeds_from_name(p)
                print(f"Migrating {p} with seeds {seeds}")
                new = _clone_into_new_esn(old, seeds)

                if overwrite:
                    save_esn(new, p)
                else:
                    save_esn(new, p + ".new.pt")

            except Exception as e:
                print("Skip", p, "->", e)
    migrate_legacy_models("./examples/Heisenberg_Chain/cache/trainedmodel_*.pt")