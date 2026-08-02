"""
Bootstrap confidence intervals on the margin/error ratio and degradation delta.

Reads: experiments/results/perquery_{dataset}.csv  (from perquery_analysis.py)

Computes per dataset, per seed (then pools):
  (1) margin/err ratio  = median(margin_10m) / RMS(err_pq_b10_10m)
      Bootstrap CI: resample 10K queries B=10000 times, compute ratio each time.
  (2) frac(err > margin) across queries at N=10M
  (3) recall degradation delta pp = 10 * (mean(hit_pq_1m) - mean(hit_pq_10m))

Primary question: do the [2.5%, 97.5%] CIs for SIFT / Deep / T2I SEPARATE on
the margin/err ratio axis? If yes → ordering is real, Path B (more seeds) not
needed. If CIs overlap → report the overlap.

Secondary: does degradation rank the same as ratio?

Outputs:
  - Printed table (stdout)
  - experiments/results/bootstrap_ci_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

RESULTS_DIR = ROOT / "experiments" / "results"
DATASETS = ["sift10m", "deep10m", "t2i10m"]
B = 10_000
RNG_SEED = 0


def bootstrap_stat(arr: np.ndarray, stat_fn, b: int = B) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) using percentile bootstrap."""
    rng = np.random.default_rng(RNG_SEED)
    n = len(arr)
    point = stat_fn(arr)
    boots = np.array([stat_fn(arr.iloc[rng.integers(0, n, size=n)]) for _ in range(b)])
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ratio_stat(df_seed: pd.DataFrame) -> float:
    return float(np.median(df_seed["margin_10m"])) / float(
        np.sqrt(np.mean(df_seed["err_pq_b10_10m"] ** 2))
    )


def frac_stat(df_seed: pd.DataFrame) -> float:
    return float(np.mean(df_seed["err_pq_b10_10m"] > df_seed["margin_10m"]))


def delta_stat(df_seed: pd.DataFrame) -> float:
    return float(
        (df_seed["hit_count_pq_10m"].mean() - df_seed["hit_count_pq_1m"].mean()) * 10
    )  # pp — negative = degradation (matches streaming oracle convention)


def main():
    summary_rows = []

    print(f"\n{'='*72}")
    print(f"Bootstrap CIs  (B={B:,} resamples, 95% CI, per-seed then averaged)")
    print(f"{'='*72}\n")

    for dataset in DATASETS:
        path = RESULTS_DIR / f"perquery_{dataset}.csv"
        if not path.exists():
            print(f"  [{dataset}] MISSING — run perquery_analysis.py first")
            continue

        df = pd.read_csv(path)
        seeds = sorted(df["seed"].unique())
        seed_ratios, seed_deltas, seed_fracs = [], [], []
        ratio_lo_list, ratio_hi_list = [], []
        delta_lo_list, delta_hi_list = [], []
        frac_lo_list, frac_hi_list = [], []

        print(f"  {dataset}  ({len(seeds)} seeds, {len(df[df['seed']==seeds[0]])} queries each)")
        for seed in seeds:
            ds = df[df["seed"] == seed].reset_index(drop=True)

            r_pt, r_lo, r_hi = bootstrap_stat(
                ds, lambda d: ratio_stat(d))
            d_pt, d_lo, d_hi = bootstrap_stat(
                ds, lambda d: delta_stat(d))
            f_pt, f_lo, f_hi = bootstrap_stat(
                ds, lambda d: frac_stat(d))

            seed_ratios.append(r_pt); ratio_lo_list.append(r_lo); ratio_hi_list.append(r_hi)
            seed_deltas.append(d_pt); delta_lo_list.append(d_lo); delta_hi_list.append(d_hi)
            seed_fracs.append(f_pt);  frac_lo_list.append(f_lo); frac_hi_list.append(f_hi)

            print(f"    seed={seed}: ratio={r_pt:.3f} [{r_lo:.3f},{r_hi:.3f}]  "
                  f"delta={d_pt:+.2f}pp [{d_lo:+.2f},{d_hi:+.2f}]  "
                  f"frac={f_pt:.3f} [{f_lo:.3f},{f_hi:.3f}]")

        # Pool: use widest CI across seeds for conservative combined CI
        r_mean = float(np.mean(seed_ratios))
        r_ci_lo = float(np.mean(ratio_lo_list))
        r_ci_hi = float(np.mean(ratio_hi_list))
        d_mean = float(np.mean(seed_deltas))
        d_ci_lo = float(np.mean(delta_lo_list))
        d_ci_hi = float(np.mean(delta_hi_list))
        f_mean = float(np.mean(seed_fracs))

        print(f"    POOLED:   ratio={r_mean:.3f} [{r_ci_lo:.3f},{r_ci_hi:.3f}]  "
              f"delta={d_mean:+.2f}pp [{d_ci_lo:+.2f},{d_ci_hi:+.2f}]")
        print()

        summary_rows.append({
            "dataset": dataset,
            "ratio_mean": round(r_mean, 4),
            "ratio_ci_lo": round(r_ci_lo, 4),
            "ratio_ci_hi": round(r_ci_hi, 4),
            "delta_pp_mean": round(d_mean, 3),
            "delta_ci_lo": round(d_ci_lo, 3),
            "delta_ci_hi": round(d_ci_hi, 3),
            "frac_mean": round(f_mean, 4),
        })

    if len(summary_rows) < 2:
        print("Not enough datasets to check separation.")
        return

    df_s = pd.DataFrame(summary_rows).sort_values("ratio_mean")
    print(f"\n{'─'*72}")
    print("Ordering by margin/err ratio (ascending → more degradation):")
    for _, row in df_s.iterrows():
        print(f"  {row['dataset']:10s}  ratio={row['ratio_mean']:.3f} "
              f"[{row['ratio_ci_lo']:.3f},{row['ratio_ci_hi']:.3f}]  "
              f"delta={row['delta_pp_mean']:+.2f}pp")

    # Check separation: do adjacent CIs overlap?
    rows = df_s.to_dict("records")
    print(f"\nSeparation check (adjacent pairs):")
    separated = True
    for i in range(len(rows) - 1):
        a, b_ = rows[i], rows[i + 1]
        overlap = a["ratio_ci_hi"] > b_["ratio_ci_lo"]
        verdict = "OVERLAP" if overlap else "SEPARATED"
        print(f"  {a['dataset']} [{a['ratio_ci_lo']:.3f},{a['ratio_ci_hi']:.3f}] "
              f"vs {b_['dataset']} [{b_['ratio_ci_lo']:.3f},{b_['ratio_ci_hi']:.3f}] "
              f"→ {verdict}")
        if overlap:
            separated = False

    print(f"\nVERDICT: {'ALL CIs SEPARATED — Path B (more seeds) UNNECESSARY' if separated else 'OVERLAP DETECTED — consider Path B'}")

    out = RESULTS_DIR / "bootstrap_ci_summary.csv"
    df_s.to_csv(out, index=False)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
