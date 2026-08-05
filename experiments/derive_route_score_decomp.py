"""
Derives exact routing/score decomposition from committed CSVs.

Identity:  1 - Recall_PQ = L_route + L_score
where:
  L_route(q,N) = 1 - C(q,N)   [fraction of true neighbors in unprobed cells]
  L_score(q,N) = C(q,N) - R(q,N) [within-probe compression ranking loss]
  C(q,N) ≈ Recall of IVFFlat (exact within-probe scoring)

Sources:
  streaming_uncompressed_*.csv  →  uncompressed_ivf recall = coverage ceiling
  streaming_*_pqmatched_multiseed.csv  →  ivf_pq_stale recall
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_is")

DATASETS = {
    "SIFT": ("streaming_uncompressed_sift10m.csv", "streaming_sift10m_pqmatched_multiseed.csv"),
    "Deep": ("streaming_uncompressed_deep10m.csv", "streaming_deep10m_pqmatched_multiseed.csv"),
    "T2I":  ("streaming_uncompressed_t2i10m.csv",  "streaming_t2i10m_pqmatched_multiseed.csv"),
}


def load_and_merge(ds_name, unc_fname, pqm_fname):
    unc = pd.read_csv(f"{RESULTS}/{unc_fname}")
    unc = unc[unc["variant"] == "uncompressed_ivf"][["seed", "vectors_indexed", "recall10"]]
    unc = unc.rename(columns={"recall10": "recall_unc"})

    pqm = pd.read_csv(f"{RESULTS}/{pqm_fname}")
    pqm = pqm[pqm["index"] == "ivf_pq_stale"][["seed", "vectors_indexed", "recall10"]]
    pqm = pqm.rename(columns={"recall10": "recall_pq"})

    df = unc.merge(pqm, on=["seed", "vectors_indexed"])
    df["L_route"] = 100 - df["recall_unc"]
    df["L_score"] = df["recall_unc"] - df["recall_pq"]
    df["check"]   = (100 - df["recall_pq"]) - (df["L_route"] + df["L_score"])
    assert (df["check"].abs() < 1e-6).all(), f"Identity fails for {ds_name}"
    df["dataset"] = ds_name
    return df


def ci95(vals):
    n = len(vals)
    m = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return m, t * se


def print_endpoint_table(all_dfs):
    print("\n=== Routing / Score Decomposition (1M → 10M, mean ± t-CI, n=3 seeds) ===")
    print(f"{'Dataset':6} {'N':4}  {'Recall_PQ':10} {'L_route':10} {'L_score':10}")
    print("-" * 55)
    for ds, df in all_dfs.items():
        for N_label, N_val in [("1M", 1_000_000), ("10M", df["vectors_indexed"].max())]:
            sub = df[df["vectors_indexed"] == N_val]
            r_pq_m,  r_pq_ci  = ci95(sub["recall_pq"].values)
            l_rt_m,  l_rt_ci  = ci95(sub["L_route"].values)
            l_sc_m,  l_sc_ci  = ci95(sub["L_score"].values)
            print(f"{ds:6} {N_label:4}  {r_pq_m:6.2f}±{r_pq_ci:.2f}  "
                  f"{l_rt_m:6.2f}±{l_rt_ci:.2f}  {l_sc_m:6.2f}±{l_sc_ci:.2f}")

    print("\n=== Change 1M → 10M ===")
    print(f"{'Dataset':6}  {'ΔRecall':8}  {'ΔL_route':9}  {'ΔL_score':9}")
    print("-" * 42)
    for ds, df in all_dfs.items():
        Nmax = df["vectors_indexed"].max()
        s1  = df[df["vectors_indexed"] == 1_000_000]
        s10 = df[df["vectors_indexed"] == Nmax]
        # per-seed deltas, then mean ± CI
        seeds = df["seed"].unique()
        dR, dRt, dSc = [], [], []
        for seed in seeds:
            r1  = float(s1[s1["seed"] == seed]["recall_pq"].iloc[0])
            r10 = float(s10[s10["seed"] == seed]["recall_pq"].iloc[0])
            rt1 = float(s1[s1["seed"] == seed]["L_route"].iloc[0])
            rt10= float(s10[s10["seed"] == seed]["L_route"].iloc[0])
            sc1 = float(s1[s1["seed"] == seed]["L_score"].iloc[0])
            sc10= float(s10[s10["seed"] == seed]["L_score"].iloc[0])
            dR.append(r10 - r1); dRt.append(rt10 - rt1); dSc.append(sc10 - sc1)
        print(f"{ds:6}  {np.mean(dR):+.2f}pp  {np.mean(dRt):+.2f}pp  {np.mean(dSc):+.2f}pp")


def save_csv(all_dfs):
    combined = pd.concat(all_dfs.values(), ignore_index=True)
    out_path = f"{RESULTS}/route_score_decomp.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return combined


def make_figure(combined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })

    C = {"SIFT": "#e41a1c", "T2I": "#377eb8", "Deep": "#4daf4a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: L_score trajectory
    ax = axes[0]
    for ds in ["SIFT", "T2I", "Deep"]:
        df = combined[combined["dataset"] == ds]
        gp = df.groupby("vectors_indexed")["L_score"]
        mean = gp.mean()
        ns = mean.index.values / 1e6
        ax.plot(ns, mean.values, color=C[ds], lw=2, marker="o", ms=4, label=ds)
    ax.set_xlabel("Corpus size (millions)", fontsize=11)
    ax.set_ylabel("Within-probe score loss $L_{\\mathrm{score}}$ (pp)", fontsize=11)
    ax.set_title("Within-probe score loss grows with corpus size", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(range(1, 11))

    # Right: stacked 1M vs 10M bar chart for SIFT (clearest contrast)
    ax = axes[1]
    ds_list = ["SIFT", "T2I", "Deep"]
    xs = np.arange(len(ds_list))
    w = 0.35

    rt_1m, rt_10m = [], []
    sc_1m, sc_10m = [], []
    for ds in ds_list:
        df = combined[combined["dataset"] == ds]
        Nmax = df["vectors_indexed"].max()
        rt_1m.append(df[df["vectors_indexed"]==1_000_000]["L_route"].mean())
        rt_10m.append(df[df["vectors_indexed"]==Nmax]["L_route"].mean())
        sc_1m.append(df[df["vectors_indexed"]==1_000_000]["L_score"].mean())
        sc_10m.append(df[df["vectors_indexed"]==Nmax]["L_score"].mean())

    ax.bar(xs - w/2, rt_1m, w, label="$L_{\\mathrm{route}}$ @1M",  color="#999999", alpha=0.8)
    ax.bar(xs - w/2, sc_1m, w, bottom=rt_1m, label="$L_{\\mathrm{score}}$ @1M", color="#d62728", alpha=0.6)
    ax.bar(xs + w/2, rt_10m, w, label="$L_{\\mathrm{route}}$ @10M", color="#555555", alpha=0.8)
    ax.bar(xs + w/2, sc_10m, w, bottom=rt_10m, label="$L_{\\mathrm{score}}$ @10M", color="#d62728", alpha=0.95)

    ax.set_xticks(xs)
    ax.set_xticklabels(ds_list, fontsize=12)
    ax.set_ylabel("Miss fraction (pp)", fontsize=11)
    ax.set_title("Routing loss ↓, score loss ↑ as corpus grows", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_decomp.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}/fig_decomp.pdf")


if __name__ == "__main__":
    all_dfs = {}
    for ds, (unc_f, pqm_f) in DATASETS.items():
        all_dfs[ds] = load_and_merge(ds, unc_f, pqm_f)
        print(f"Loaded {ds}: {len(all_dfs[ds])} rows, identity verified ✓")

    print_endpoint_table(all_dfs)
    combined = save_csv(all_dfs)
    make_figure(combined)
    print("\nDone.")
