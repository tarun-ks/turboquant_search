"""
CLI entry point for TurboQuant Search.

Usage:
    tqs benchmark --dataset sift-1m       # Run benchmarks, print table
    tqs index --input vectors.npy --bits 3  # Index custom embeddings
    tqs search --index tq.index --query "text" --top 10
"""

import sys
import click
import numpy as np


@click.group()
@click.version_option(package_name="turboquant-search")
def cli():
    """TurboQuant Search — vector compression for similarity search."""
    pass


@cli.command()
@click.option(
    "--dataset", "-d",
    default="synthetic",
    type=click.Choice(["synthetic", "sift-128", "glove-100", "sift-1m"]),
    help="Dataset for benchmarks.",
)
@click.option("--n-vectors", "-n", default=10000, help="Number of vectors (synthetic only).")
@click.option("--bits", "-b", multiple=True, type=int, default=(2, 3, 4), help="Bit widths to test.")
def benchmark(dataset, n_vectors, bits):
    """Run benchmarks and print results table."""
    from .benchmarks import run_benchmark, format_results_table

    bits = list(bits)
    click.echo(f"Running benchmark: {dataset} ({n_vectors:,} vectors, bits={bits})")
    click.echo()

    def progress_cb(step, total, msg):
        click.echo(f"  [{step}/{total}] {msg}")

    results = run_benchmark(
        dataset_name=dataset,
        n_vectors=n_vectors,
        n_queries=200,
        k_values=[1, 5, 10, 50],
        bit_widths=bits,
        progress_callback=progress_cb,
    )

    click.echo()
    click.echo(format_results_table(results))


@cli.command("index")
@click.option("--input", "-i", "input_path", required=True, help="Path to .npy embeddings file.")
@click.option("--bits", "-b", default=3, type=click.IntRange(2, 4), help="Quantization bits (2-4).")
@click.option("--output", "-o", default=None, help="Output index path (default: <input>.tqindex).")
@click.option("--flat", is_flag=True, default=False,
              help="Use flat TurboQuant index (prototype path; no IVF coarse partition). "
                   "Default builds IVF-TQ, the paper's main contribution.")
@click.option("--nlist", default=None, type=int,
              help="IVF coarse partitions (default: auto, ~sqrt(N) capped at 1000). Ignored with --flat.")
@click.option("--nprobe", default=10, type=int,
              help="IVF search budget (default: 10). Ignored with --flat.")
@click.option("--seed", default=42, type=int, help="Random seed for the rotation matrix (default: 42).")
def index_cmd(input_path, bits, output, flat, nlist, nprobe, seed):
    """Index custom numpy embeddings.

    Default builds an IVF-TQ index (the paper's main contribution). Use --flat to build
    the prototype flat TurboQuant index instead (suitable for small corpora up to ~100K
    or for didactic comparison).
    """
    import math
    import pickle

    click.echo(f"Loading vectors from {input_path}...")
    vectors = np.load(input_path).astype(np.float32)
    n, dim = vectors.shape
    click.echo(f"  {n:,} vectors, dim={dim}")

    if flat:
        from .core import TurboQuantSearchIndex
        click.echo(f"Building flat TQ {bits}-bit index (prototype path)...")
        idx = TurboQuantSearchIndex(dim=dim, bits=bits, seed=seed)
        idx.add(vectors)
        index_type = "flat"
    else:
        from .core import IVFTurboQuantIndex
        if nlist is None:
            nlist = max(10, min(1000, int(math.sqrt(n))))
        nprobe_eff = min(nprobe, nlist)
        click.echo(f"Building IVF-TQ {bits}-bit index "
                   f"(nlist={nlist}, nprobe={nprobe_eff}, sign-bit refinement on)...")
        idx = IVFTurboQuantIndex(
            dim=dim, nlist=nlist, bits=bits, nprobe=nprobe_eff,
            use_residual_sign=True, seed=seed,
        )
        idx.train(vectors)
        idx.add(vectors)
        index_type = "ivf"

    stats = idx.stats()
    click.echo(f"  Compression: {stats['compression_ratio']}")
    click.echo(f"  Memory: {stats['memory_mb']:.2f} MB")
    click.echo(f"  Build time: {idx.build_time:.3f}s")

    if output is None:
        output = input_path.rsplit(".", 1)[0] + ".tqindex"

    with open(output, "wb") as f:
        pickle.dump({
            "index": idx, "dim": dim, "bits": bits, "n_vectors": n,
            "index_type": index_type, "version": 2,
        }, f)
    click.echo(f"Saved {index_type} index to {output}")


@cli.command("search")
@click.option("--index", "-i", "index_path", required=True, help="Path to .tqindex file.")
@click.option("--query", "-q", required=True, help="Path to query .npy file (one or more vectors).")
@click.option("--top", "-k", default=10, help="Number of results to return.")
def search_cmd(index_path, query, top):
    """Search an existing index."""
    import pickle

    click.echo(f"Loading index from {index_path}...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    idx = data["index"]
    index_type = data.get("index_type", "flat")  # backward-compat: pre-v2 indexes were always flat
    click.echo(f"  {data['n_vectors']:,} vectors, dim={data['dim']}, {data['bits']}-bit ({index_type})")

    query_vectors = np.load(query).astype(np.float32)
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)

    click.echo(f"Searching for top-{top} results ({query_vectors.shape[0]} queries)...")
    scores, indices = idx.search(query_vectors, k=top)

    for i in range(query_vectors.shape[0]):
        click.echo(f"\nQuery {i}:")
        for rank, (score, idx_val) in enumerate(zip(scores[i], indices[i])):
            click.echo(f"  {rank+1:>3}. index={idx_val:<8} score={score:.4f}")


def main():
    cli()


if __name__ == "__main__":
    main()
