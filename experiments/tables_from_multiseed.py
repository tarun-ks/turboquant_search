"""
Generate LaTeX tables from the multi-seed streaming CSVs.

Given the CSVs produced by streaming_multiseed.py, this script computes:
  * Per-cell mean ± 95% CI (t-distribution, df = n_seeds - 1)
  * Paired-t-test differences between matched-seed comparisons
  * LaTeX table fragments drop-in-ready for the PVLDB paper

Outputs (under experiments/results/):
    table2_streaming_sift1m_controls.tex
    table12_streaming_deep10m.tex
    table13_streaming_sift10m.tex
    table_pqhigh_appendix.tex
    multiseed_summary.txt  (human-readable diagnostic)

Usage:
    python tables_from_multiseed.py        # regenerates all tables
    python tables_from_multiseed.py --csv  # also print raw cells to stdout
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"


# ── stats helpers ────────────────────────────────────────────────

def mean_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (mean, std_sample, ci_half_width) under t-distribution with df = n-1."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0, float("nan")
    std = float(np.std(values, ddof=1))
    crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci_half = crit * std / np.sqrt(n)
    return mean, std, ci_half


def paired_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Paired t-test on (a - b). Returns (mean_diff, ci_half_of_diff, p_value)."""
    assert len(a) == len(b)
    diff = a - b
    n = len(diff)
    mean_d = float(np.mean(diff))
    if n < 2:
        return mean_d, float("nan"), float("nan")
    std_d = float(np.std(diff, ddof=1))
    crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_half = crit * std_d / np.sqrt(n)
    _, p = stats.ttest_rel(a, b)
    return mean_d, ci_half, float(p)


def fmt_pct(mean: float, ci: float) -> str:
    """Format as '87.52 ± 0.18' (no % sign; the column header carries the unit)."""
    if np.isnan(ci):
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {ci:.2f}"


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "$p<0.001$"
    return f"$p={p:.3f}$"


def fmt_diff(mean: float, ci: float, p: float) -> str:
    sign = "+" if mean >= 0 else ""
    return f"${sign}{mean:.2f} \\pm {ci:.2f}$ pp, {fmt_p(p)}"


# ── memory accounting ────────────────────────────────────────────
# IVF-TQ storage (from turboquant_search/core.py:836-842):
#   bits/vec = bits_lloyd * dim + dim (sign bit) + 32 (norm float32)
#            = (bits_lloyd + 1) * dim + 32
# IVF-PQ storage: bits/vec = m_pq * bits_per_sub
# All TQ runs use bits_lloyd=4 with use_residual_sign=True, so 5*dim + 32.

_DATASET_DIMS = {"deep10m": 96, "sift10m": 128, "sift1m": 128}
_TQ_BITS_LLOYD = 4

def bits_per_vector(index_name: str, dataset: str,
                    m_pq: int = 0, bits_per_sub: int = 0) -> int:
    dim = _DATASET_DIMS.get(dataset, 128)
    if index_name == "ivf_tq":
        return (_TQ_BITS_LLOYD + 1) * dim + 32
    if index_name in ("ivf_pq_stale", "ivf_pq_retrain", "ivf_pq"):
        return int(m_pq) * int(bits_per_sub)
    return 0


# ── Table 12 / 13 / pqhigh (10M streaming) ───────────────────────

def build_10m_summary(csv_path: Path, dataset_label: str,
                      caption: str, label: str) -> Tuple[str, str]:
    """Build a 1M→10M summary table for one 10M streaming CSV.

    Returns (latex, human_summary).
    """
    df = pd.read_csv(csv_path)
    seeds = sorted(df["seed"].unique())
    indexes = ["ivf_tq", "ivf_pq_stale", "ivf_pq_retrain"]
    index_labels = {
        "ivf_tq": "IVF-TQ",
        "ivf_pq_stale": "IVF-PQ stale",
        "ivf_pq_retrain": "IVF-PQ retrain",
    }
    final_state = int(df["vectors_indexed"].max())
    dataset_key = str(df["dataset"].iloc[0])
    m_pq = int(df["m_pq"].iloc[0])
    bps = int(df["bits_per_sub"].iloc[0])
    tq_bits = bits_per_vector("ivf_tq", dataset_key)
    pq_bits = bits_per_vector("ivf_pq_stale", dataset_key, m_pq, bps)
    bits_for_index = {
        "ivf_tq": tq_bits,
        "ivf_pq_stale": pq_bits,
        "ivf_pq_retrain": pq_bits,
    }

    def cell(index_name: str, n_indexed: int) -> np.ndarray:
        sub = df[(df["index"] == index_name) & (df["vectors_indexed"] == n_indexed)]
        return sub.sort_values("seed")["recall10"].values

    rows_latex: List[str] = []
    for ix in indexes:
        vals_1m = cell(ix, 1_000_000)
        vals_10m = cell(ix, final_state)
        m1, _, c1 = mean_ci(vals_1m)
        m10, _, c10 = mean_ci(vals_10m)
        d_mean, d_ci, d_p = paired_test(vals_10m, vals_1m)
        rows_latex.append(
            f"  {dataset_label} & {index_labels[ix]} & {bits_for_index[ix]} & "
            f"{fmt_pct(m1, c1)} & {fmt_pct(m10, c10)} & "
            f"{fmt_diff(d_mean, d_ci, d_p)} \\\\"
        )

    # Cross-index comparisons at final-state (10M for SIFT-10M, 9.99M for Deep-10M)
    tq_10m = cell("ivf_tq", final_state)
    pq_s_10m = cell("ivf_pq_stale", final_state)
    pq_r_10m = cell("ivf_pq_retrain", final_state)
    tq_vs_pq_s = paired_test(tq_10m, pq_s_10m)
    pq_s_vs_r = paired_test(pq_s_10m, pq_r_10m)

    latex = (
        f"\\begin{{table}}[t]\n"
        f"\\caption{{{caption} Memory: IVF-TQ {tq_bits} bits/vec, IVF-PQ {pq_bits} bits/vec "
        f"({pq_bits/tq_bits:.2f}$\\times$ IVF-TQ).}}\n"
        f"\\label{{{label}}}\n"
        f"\\centering\\small\n"
        f"\\begin{{tabular}}{{llcccc}}\n"
        f"\\toprule\n"
        f"Dataset & Index & Bits/vec & R@10 (1M) & R@10 (10M) & Change $\\Delta$ \\\\\n"
        f"\\midrule\n"
        + "\n".join(rows_latex) + "\n"
        f"\\midrule\n"
        f"\\multicolumn{{6}}{{l}}{{\\textit{{IVF-TQ vs.\\ IVF-PQ stale at 10M: "
        f"{fmt_diff(*tq_vs_pq_s)}}}}} \\\\\n"
        f"\\multicolumn{{6}}{{l}}{{\\textit{{IVF-PQ stale vs.\\ retrain at 10M: "
        f"{fmt_diff(*pq_s_vs_r)}}}}} \\\\\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )

    summary_lines = [
        f"=== {dataset_label} ({csv_path.name}) ===",
        f"seeds: {seeds}",
    ]
    for ix in indexes:
        vals_1m = cell(ix, 1_000_000)
        vals_10m = cell(ix, final_state)
        summary_lines.append(
            f"  {index_labels[ix]}: "
            f"1M={mean_ci(vals_1m)[0]:.2f}±{mean_ci(vals_1m)[2]:.2f} | "
            f"10M={mean_ci(vals_10m)[0]:.2f}±{mean_ci(vals_10m)[2]:.2f} | "
            f"Δ={paired_test(vals_10m, vals_1m)[0]:+.2f}pp "
            f"(p={paired_test(vals_10m, vals_1m)[2]:.3f})"
        )
    summary_lines.append(f"  IVF-TQ vs IVF-PQ stale at 10M: "
                         f"Δ={tq_vs_pq_s[0]:+.2f}±{tq_vs_pq_s[1]:.2f}pp "
                         f"(p={tq_vs_pq_s[2]:.3f})")
    summary_lines.append(f"  IVF-PQ stale vs retrain at 10M: "
                         f"Δ={pq_s_vs_r[0]:+.2f}±{pq_s_vs_r[1]:.2f}pp "
                         f"(p={pq_s_vs_r[2]:.3f})")

    return latex, "\n".join(summary_lines)


# ── Table 2 (SIFT-1M ingestion controls) ─────────────────────────

def build_sift1m_controls(csv_path: Path,
                          caption: str, label: str) -> Tuple[str, str]:
    df = pd.read_csv(csv_path)
    conditions = ["original", "shuffled", "mean_shift"]
    cond_labels = {
        "original": "Original order",
        "shuffled": "Shuffled (i.i.d.)",
        "mean_shift": "Mean-shift (0.05/batch)",
    }
    indexes = ["ivf_tq", "ivf_pq"]
    states = ["200K", "1M"]

    def cell(cond: str, ix: str, st: str) -> np.ndarray:
        sub = df[(df["condition"] == cond) & (df["index"] == ix) & (df["state"] == st)]
        return sub.sort_values("seed")["recall10"].values

    rows_latex: List[str] = []
    summary_lines: List[str] = []
    for cond in conditions:
        tq_200k = cell(cond, "ivf_tq", "200K")
        tq_1m   = cell(cond, "ivf_tq", "1M")
        pq_200k = cell(cond, "ivf_pq", "200K")
        pq_1m   = cell(cond, "ivf_pq", "1M")
        m_tq_200, _, c_tq_200 = mean_ci(tq_200k)
        m_tq_1m,  _, c_tq_1m  = mean_ci(tq_1m)
        m_pq_200, _, c_pq_200 = mean_ci(pq_200k)
        m_pq_1m,  _, c_pq_1m  = mean_ci(pq_1m)
        tq_change = paired_test(tq_1m, tq_200k)
        pq_change = paired_test(pq_1m, pq_200k)
        rows_latex.append(
            f"  {cond_labels[cond]} & "
            f"{fmt_pct(m_tq_200, c_tq_200)} $\\to$ {fmt_pct(m_tq_1m, c_tq_1m)} & "
            f"{fmt_pct(m_pq_200, c_pq_200)} $\\to$ {fmt_pct(m_pq_1m, c_pq_1m)} & "
            f"{fmt_diff(*tq_change)} & {fmt_diff(*pq_change)} \\\\"
        )
        summary_lines.append(
            f"  {cond_labels[cond]}: "
            f"TQ {m_tq_200:.2f}→{m_tq_1m:.2f} "
            f"(Δ={tq_change[0]:+.2f}, p={tq_change[2]:.3f}); "
            f"PQ {m_pq_200:.2f}→{m_pq_1m:.2f} "
            f"(Δ={pq_change[0]:+.2f}, p={pq_change[2]:.3f})"
        )

    # SIFT-1M memory accounting (hardcoded in orchestrator: m_pq=64, bits_per_sub=8 PQ;
    # IVF-TQ bits_lloyd=4 with use_residual_sign=True on dim=128).
    tq_bits = bits_per_vector("ivf_tq", "sift1m")
    pq_bits = bits_per_vector("ivf_pq", "sift1m", m_pq=64, bits_per_sub=8)
    mem_note = (
        f" Memory: IVF-TQ {tq_bits} bits/vec, "
        f"IVF-PQ {pq_bits} bits/vec ({pq_bits/tq_bits:.2f}$\\times$ IVF-TQ)."
    )
    latex = (
        f"\\begin{{table}}[t]\n"
        f"\\caption{{{caption}{mem_note}}}\n"
        f"\\label{{{label}}}\n"
        f"\\centering\\small\n"
        f"\\begin{{tabular}}{{lcccc}}\n"
        f"\\toprule\n"
        f"\\textbf{{Condition}} & \\textbf{{IVF-TQ (200K$\\to$1M)}} & "
        f"\\textbf{{IVF-PQ (200K$\\to$1M)}} & "
        f"\\textbf{{TQ $\\Delta$}} & \\textbf{{PQ $\\Delta$}} \\\\\n"
        f"\\midrule\n"
        + "\n".join(rows_latex) + "\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    return latex, "=== SIFT-1M controls ===\n" + "\n".join(summary_lines)


# ── Main ─────────────────────────────────────────────────────────

JOBS = [
    {
        "csv": "streaming_sift1m_multiseed.csv",
        "builder": "sift1m",
        "out_tex": "table2_streaming_sift1m_controls.tex",
        "caption": (
            "SIFT-1M streaming under three ingestion conditions, 3 seeds "
            "(42, 123, 7777). Recall@10 reported as mean $\\pm$ 95\\% CI "
            "(t-distribution, df=2). Paired t-tests on within-seed differences."
        ),
        "label": "tab:streaming_controls_main",
    },
    {
        "csv": "streaming_deep10m_multiseed.csv",
        "builder": "10m",
        "dataset_label": "Deep-10M",
        "out_tex": "table12_streaming_deep10m.tex",
        "caption": (
            "Streaming on Deep-10M, 3 seeds. Mean $\\pm$ 95\\% CI; paired "
            "t-tests on within-seed differences."
        ),
        "label": "tab:streaming_deep10m_multiseed",
    },
    {
        "csv": "streaming_sift10m_multiseed.csv",
        "builder": "10m",
        "dataset_label": "SIFT-10M",
        "out_tex": "table13_streaming_sift10m.tex",
        "caption": (
            "Streaming on SIFT-10M, 3 seeds. Mean $\\pm$ 95\\% CI; paired "
            "t-tests on within-seed differences."
        ),
        "label": "tab:streaming_sift10m_multiseed",
    },
    {
        "csv": "streaming_deep10m_pqhigh_multiseed.csv",
        "builder": "10m",
        "dataset_label": "Deep-10M (PQ $m{=}96$, 8-bit)",
        "out_tex": "table_pqhigh_deep10m.tex",
        "caption": (
            "Higher PQ bit budget on Deep-10M (super-matched memory): IVF-PQ at "
            "$m{=}96$, 8 bits/subspace (vs.\\ $m{=}48$ in "
            "Table~\\ref{tab:streaming_deep10m_multiseed}). 3 seeds; mean $\\pm$ 95\\% CI."
        ),
        "label": "tab:streaming_deep10m_pqhigh",
    },
    {
        "csv": "streaming_sift10m_pqhigh_multiseed.csv",
        "builder": "10m",
        "dataset_label": "SIFT-10M (PQ $m{=}128$, 8-bit)",
        "out_tex": "table_pqhigh_sift10m.tex",
        "caption": (
            "Higher PQ bit budget on SIFT-10M (super-matched memory): IVF-PQ at "
            "$m{=}128$, 8 bits/subspace (vs.\\ $m{=}64$ in "
            "Table~\\ref{tab:streaming_sift10m_multiseed}). 3 seeds; mean $\\pm$ 95\\% CI."
        ),
        "label": "tab:streaming_sift10m_pqhigh",
    },
    {
        "csv": "streaming_deep10m_pqmatched_multiseed.csv",
        "builder": "10m",
        "dataset_label": "Deep-10M (PQ $m{=}48$, 10-bit)",
        "out_tex": "table_pqmatched_deep10m.tex",
        "caption": (
            "Bit-matched PQ on Deep-10M: IVF-PQ at $m{=}48$, 10 bits/subspace "
            "(480 bits/vec, $\\sim$0.94$\\times$ IVF-TQ memory). 3 seeds; mean $\\pm$ 95\\% CI."
        ),
        "label": "tab:streaming_deep10m_pqmatched",
    },
    {
        "csv": "streaming_sift10m_pqmatched_multiseed.csv",
        "builder": "10m",
        "dataset_label": "SIFT-10M (PQ $m{=}64$, 10-bit)",
        "out_tex": "table_pqmatched_sift10m.tex",
        "caption": (
            "Bit-matched PQ on SIFT-10M: IVF-PQ at $m{=}64$, 10 bits/subspace "
            "(640 bits/vec, $\\sim$0.95$\\times$ IVF-TQ memory). 3 seeds; mean $\\pm$ 95\\% CI."
        ),
        "label": "tab:streaming_sift10m_pqmatched",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dump", action="store_true",
                    help="Also print per-row CSV content to stdout.")
    args = ap.parse_args()

    summaries: List[str] = []
    for job in JOBS:
        csv_path = RESULTS_DIR / job["csv"]
        if not csv_path.exists():
            print(f"skipping {job['csv']}: not yet produced")
            continue
        if job["builder"] == "sift1m":
            tex, summary = build_sift1m_controls(csv_path, job["caption"], job["label"])
        else:
            tex, summary = build_10m_summary(
                csv_path, job["dataset_label"], job["caption"], job["label"])
        out_tex = RESULTS_DIR / job["out_tex"]
        out_tex.write_text(tex)
        print(f"wrote {out_tex}")
        summaries.append(summary)
        if args.csv_dump:
            print(pd.read_csv(csv_path).to_string())

    summary_path = RESULTS_DIR / "multiseed_summary.txt"
    summary_path.write_text("\n\n".join(summaries) + "\n")
    print(f"\nhuman summary: {summary_path}")


if __name__ == "__main__":
    main()
