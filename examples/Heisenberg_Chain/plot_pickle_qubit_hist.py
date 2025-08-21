"""
One-stop workflow:
  - Generate many TrueHist_*.pkl files (using your HeisenbergChain method + cache checks)
  - Combine them into one histogram per qubit (frequency vs value) across all files

Notes
-----
- This script expects your repo in PYTHONPATH / run it from your repo so HeisenbergChain imports cleanly.
- It reproduces your cache behavior: if an output file exists and `N`, `dt`, and `op` match, it reuses it.

Output
------
- Generation: TrueHist_Seed{seed}_N{N}_qubits({qmin}-{qmax})_op({op})_dt{dt_underscore}.pkl in OUTPUT_DIR
- Combination: combined_hist_q{q}.png and combined_summary.csv (and combined_counts_q{q}.csv for discrete-like data)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from typing import Sequence, Dict, Tuple, Any
from .Heisenberg_sim import HeisenbergChain

# ------------------------- HeisenbergChain import helper -------------------------


# ------------------------- File naming & generation -------------------------
def truth_cache_name(base_dir: str, seed: int, N: int, qubits: Sequence[int], op: str, dt: float) -> str:
    qmin, qmax = min(qubits), max(qubits)
    dt_tag = str(dt).replace(".", "_", 1)
    fname = f"TrueHist_Seed{seed}_N{N}_qubits({qmin}-{qmax})_op({op})_dt{dt_tag}.pkl"
    return os.path.join(base_dir, fname)


def generate_one(
    base_dir: str,
    seed: int,
    N: int,
    qubits: Sequence[int],
    op: str,
    dt: float,
    T: int,
    HeisenbergChain
) -> str:
    """
    Generate (or reuse) one TrueHist_*.pkl for the given seed.
    Returns the path to the pickle.
    """
    path = truth_cache_name(base_dir, seed, N, qubits, op, dt)
    steps = int(T / dt)

    # Reuse cached file if metadata match
    if os.path.exists(path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("N") == N and payload.get("dt") == dt and payload.get("op") == op:
            print(f"[truth-cache] Loaded {path}")
            return path

    print(f"[truth-cache] Generating true series for N={N}, qubits={list(qubits)}, op={op}, dt={dt}, seed={seed}")
    series: Dict[int, np.ndarray] = {}
    for q in qubits:
        np.random.seed(seed)
        chain = HeisenbergChain(num_qubits=N, target_qubit=q, dt=dt, measure=op)
        chain.evolve(steps)
        series[q] = np.asarray(chain.get_observable(), dtype=float).ravel()

    payload = {"N": N, "dt": dt, "T": T, "op": op, "seed": seed, "qubits": list(qubits), "series": series}
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[truth-cache] Saved -> {path}")
    return path


def generate_many(
    out_dir: str,
    N: int,
    qubits: Sequence[int],
    op: str,
    dt: float,
    T: int,
    seeds: int,
    seed_start: int,
    HeisenbergChain
):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(seeds):
        seed = seed_start + i
        generate_one(out_dir, seed, N, qubits, op, dt, T, HeisenbergChain)


# ------------------------- Combination (histograms across many files) -------------------------
def load_series(path: str) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    meta = {k: payload.get(k) for k in ["N", "dt", "op", "T", "seed", "qubits"]}
    series = payload.get("series", {})
    normalized: Dict[int, np.ndarray] = {}
    for k, v in series.items():
        # normalize qubit key to int
        try:
            ik = int(k)
        except Exception:
            s = str(k)
            digits = "".join(ch for ch in s if ch.isdigit())
            ik = int(digits) if digits else k
        a = np.asarray(v).ravel().astype(float)
        normalized[ik] = a
    return meta, normalized


def is_discrete_like(arr: np.ndarray) -> bool:
    if np.issubdtype(arr.dtype, np.integer):
        return True
    rounded = np.round(arr, 12)
    return np.unique(rounded).size <= 50


def histogram_bins(arr: np.ndarray, discrete: bool):
    if discrete:
        vals = np.unique(np.round(arr, 12)).astype(float)
        if vals.size == 1:
            v = float(vals[0])
            eps = 0.5 if float(v).is_integer() else 1e-6
            return np.array([v - eps, v + eps])
        mids = (vals[:-1] + vals[1:]) / 2.0
        edges = np.concatenate(([vals[0] - (mids[0] - vals[0])], mids, [vals[-1] + (vals[-1] - mids[-1])]))
        return edges
    return "auto"


def combine_histograms(
    src_dir: str,
    out_dir: str,
    expect_N: int = None,
    expect_op: str = None,
    expect_dt: float = None,
    skip_checks: bool = False
):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob(os.path.join(src_dir, "TrueHist_*.pkl")))
    if not paths:
        print(f"[combine] No TrueHist_*.pkl files found in {src_dir}")
        return

    combined: Dict[int, list] = defaultdict(list)
    used_files = 0

    for p in paths:
        try:
            meta, series = load_series(p)
        except Exception as e:
            print(f"[combine][skip] {p}: load error: {e}")
            continue

        if not skip_checks:
            if (expect_N is not None and meta.get("N") != expect_N) or \
               (expect_op is not None and meta.get("op") != expect_op) or \
               (expect_dt is not None and abs(meta.get("dt") - expect_dt) > 1e-12):
                print(f"[combine][skip] {p}: metadata mismatch (got N={meta.get('N')}, op={meta.get('op')}, dt={meta.get('dt')})")
                continue

        for q, arr in series.items():
            combined[q].append(arr)
        used_files += 1

    if used_files == 0:
        print("[combine] No files matched expected metadata; nothing to combine.")
        return

    rows = []
    for q in sorted(combined.keys()):
        data = np.concatenate(combined[q], axis=0)
        discrete = is_discrete_like(data)
        bins = histogram_bins(data, discrete)

        plt.figure()
        plt.hist(data, bins=bins)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(f"Combined Histogram q{q} across {used_files} files (n={data.size})")
        out_png = os.path.join(out_dir, f"combined_hist_q{q}.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=144)
        plt.close()

        rows.append({
            "qubit": f"q{q}",
            "files": used_files,
            "n_total": int(data.size),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "discrete_like": bool(discrete),
        })

        if discrete:
            rounded = np.round(data, 12)
            vals, counts = np.unique(rounded, return_counts=True)
            pd.DataFrame({"value": vals.astype(float), "count": counts.astype(int)}).to_csv(
                os.path.join(out_dir, f"combined_counts_q{q}.csv"), index=False
            )

    pd.DataFrame(rows).sort_values("qubit").to_csv(os.path.join(out_dir, "combined_summary.csv"), index=False)
    print(f"[combine] Combined {used_files} files -> {out_dir}")


# ------------------------- One-call workflow (optional) -------------------------
def run_workflow(
    do_generate: bool,
    out_dir: str,
    N: int,
    qubits: Sequence[int],
    op: str,
    dt: float,
    T: int,
    seeds: int,
    seed_start: int,
    combine_src_dir: str,
    combine_out_dir: str,
    expect_N: int = None,
    expect_op: str = None,
    expect_dt: float = None,
    skip_checks: bool = False
):
    if do_generate:
        print("[workflow] Generating files...")
        generate_many(out_dir, N, qubits, op, dt, T, seeds, seed_start, HeisenbergChain)
    print("[workflow] Combining histograms...")
    combine_histograms(combine_src_dir, combine_out_dir, expect_N, expect_op, expect_dt, skip_checks)
    print("[workflow] Done.")


# ------------------------- Set your parameters here and run -------------------------
if __name__ == "__main__":
    # ---------- GENERATION SETTINGS ----------
    DO_GENERATE   = True        # Set False if you only want to combine existing files
    OUTPUT_DIR    = "./true_hist"  # Where to write TrueHist_*.pkl
    N             = 3
    QUBITS        = [0, 1]
    OP            = "sz"
    DT            = 0.2
    T             = 100            # total time; steps = int(T/DT)
    SEEDS         = 3           # how many files
    SEED_START    = 314         # first seed

    # ---------- COMBINATION SETTINGS ----------
    SRC_DIR           = OUTPUT_DIR             # folder with the TrueHist_*.pkl files
    COMBINED_OUT_DIR  = "./combined_hists"     # where to write combined histograms/CSVs
    EXPECT_N          = N                      # metadata checks; set to None to ignore
    EXPECT_OP         = OP
    EXPECT_DT         = DT
    SKIP_CHECKS       = True             # if True, combine even if metadata differ

    # ---------- RUN ----------
    run_workflow(
        do_generate=DO_GENERATE,
        out_dir=OUTPUT_DIR,
        N=N,
        qubits=QUBITS,
        op=OP,
        dt=DT,
        T=T,
        seeds=SEEDS,
        seed_start=SEED_START,
        combine_src_dir=SRC_DIR,
        combine_out_dir=COMBINED_OUT_DIR,
        expect_N=EXPECT_N,
        expect_op=EXPECT_OP,
        expect_dt=EXPECT_DT,
        skip_checks=SKIP_CHECKS,
    )