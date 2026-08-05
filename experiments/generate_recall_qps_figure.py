"""
Generate recall-QPS Pareto figure from recall_qps_results.json.
Saves paper/tex/fig_recall_qps.pdf.

Usage: python experiments/generate_recall_qps_figure.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "recall_qps_results.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "paper", "tex", "fig_recall_qps.pdf")

COLOR = {
    "flat_tq_b4":      "#999999",
    "ivf_tq_b4":       "#d62728",
    "ivf_tq_b5":       "#ff7f0e",
    "ivf_tq_b6":       "#e377c2",
    "ivf_pq_m64":      "#1f77b4",
    "ivf_pq_m128":     "#aec7e8",
    "opq_ivf_pq_m128": "#17becf",
    "opq_ivf_pq_m96":  "#17becf",
    "hnsw_m32":        "#2ca02c",
    "ext_rabitq_B5":   "#9467bd",
    "ext_rabitq_B6":   "#c5b0d5",
}
LABEL = {
    "flat_tq_b4":      "Flat TQ (exhaustive)",
    "ivf_tq_b4":       "IVF-TQ 4-bit (ours)",
    "ivf_tq_b5":       "IVF-TQ 5-bit (ours)",
    "ivf_tq_b6":       "IVF-TQ 6-bit (ours)",
    "ivf_pq_m64":      "FAISS IVF-PQ m=64",
    "ivf_pq_m128":     "FAISS IVF-PQ m=128",
    "opq_ivf_pq_m128": "FAISS OPQ+IVF-PQ m=128",
    "opq_ivf_pq_m96":  "FAISS OPQ+IVF-PQ m=96",
    "hnsw_m32":        "FAISS HNSW M=32",
    "ext_rabitq_B5":   "Ext. RaBitQ B=5",
    "ext_rabitq_B6":   "Ext. RaBitQ B=6",
}
MARKER = {
    "flat_tq_b4":      "x",
    "ivf_tq_b4":       "o",
    "ivf_tq_b5":       "s",
    "ivf_tq_b6":       "^",
    "ivf_pq_m64":      "D",
    "ivf_pq_m128":     "D",
    "opq_ivf_pq_m128": "P",
    "opq_ivf_pq_m96":  "P",
    "hnsw_m32":        "v",
    "ext_rabitq_B5":   "*",
    "ext_rabitq_B6":   "*",
}
LINEWIDTH = {k: (2.5 if "ivf_tq" in k else 1.5) for k in COLOR}
ZORDER = {k: (5 if "ivf_tq" in k else 3) for k in COLOR}

def extract_pts(data, key):
    """Extract (recall10, qps) pairs from a method entry."""
    entry = data.get(key)
    if entry is None:
        return [], []
    if isinstance(entry, dict):
        # Single point (flat search)
        return [entry["recall10"]], [entry["qps"]]
    # List of nprobe / ef_search sweep points
    pts = sorted(entry, key=lambda x: x["recall10"])
    return [p["recall10"] for p in pts], [p["qps"] for p in pts]


def plot_dataset(ax, ds_data, ds_name, show_legend=True):
    plotted = {}
    for key in ["flat_tq_b4", "ext_rabitq_B5", "ext_rabitq_B6",
                "ivf_pq_m64", "ivf_pq_m128", "opq_ivf_pq_m128", "opq_ivf_pq_m96",
                "hnsw_m32",
                "ivf_tq_b4", "ivf_tq_b5", "ivf_tq_b6"]:
        r, q = extract_pts(ds_data, key)
        if not r:
            continue
        c = COLOR.get(key, "#333333")
        lbl = LABEL.get(key, key)
        mk = MARKER.get(key, "o")
        lw = LINEWIDTH.get(key, 1.5)
        zo = ZORDER.get(key, 3)
        if len(r) == 1:
            ax.scatter(r, q, color=c, marker=mk, s=80, zorder=zo, label=lbl)
        else:
            ax.plot(r, q, color=c, marker=mk, markersize=5, linewidth=lw,
                    zorder=zo, label=lbl)
        plotted[key] = True

    ax.set_xlabel("Recall@10 (%)", fontsize=10)
    ax.set_ylabel("QPS", fontsize=10)
    ax.set_title(ds_name, fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(left=60)
    if show_legend:
        ax.legend(fontsize=7, loc="lower right", ncol=1, framealpha=0.8)


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: {RESULTS_PATH} not found. Run recall_qps_sweep.py first.")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    datasets = list(results.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.2))
    if len(datasets) == 1:
        axes = [axes]

    for i, ds in enumerate(datasets):
        plot_dataset(axes[i], results[ds], ds, show_legend=(i == len(datasets) - 1))

    plt.tight_layout(pad=1.2)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
