# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-08-02

### Fixed
- `IVFTurboQuantIndex.memory_bytes` and `TurboQuantSearchIndex.memory_bytes`
  now documented as theoretical packed footprint (bits-perfect, no alignment
  padding). Actual numpy resident memory is substantially larger.
- `freq_adaptive.py`: corrected O(d) complexity claim to O(d²); the dense
  random rotation matrix multiply is O(d²) per vector, not O(d).
- `stats()` key `memory_mb` renamed to `memory_packed_mb`; `memory_resident_mb`
  added alongside it.
- README Limitations: updated QPS figure to confirmed ~13K at nprobe=20
  (NEON C++ kernel, SIFT-1M); prior ~22K was incorrect.
- README: two instances of "the paper" removed; wording is now self-contained.

### Added
- `memory_bytes_resident()` method on both `IVFTurboQuantIndex` and
  `TurboQuantSearchIndex` — returns the sum of `.nbytes` across all live numpy
  arrays (partition indices, sign bits, norms, codes, raw vectors).
- `experiments/measure_memory.py` — builds a SIFT-1M index and prints packed
  theoretical vs resident memory side-by-side for b=4 and b=6.
- Streaming benchmark scripts: `streaming_oracle.py`,
  `streaming_uncompressed.py`, `streaming_rerank.py`, `rank_margin.py`,
  `perquery_analysis.py`, `causal_miss.py`, `bootstrap_ci.py`,
  `qps_benchmark.py`.
- Result CSVs for SIFT-10M, Deep-10M, T2I-10M (3 seeds each) under
  `experiments/results/`.
- README: memory table column labeled as packed theoretical with pointer to
  `measure_memory.py`; rotation O(d²) complexity note in "How It Works";
  "Reproducing corpus-growth streaming results" section with exact commands,
  expected runtimes, and output files.
- `setup.py`: use `-mcpu=apple-m1` instead of `-march=native` on Apple
  Silicon for portable builds across all M-series chips.

## [0.2.0] - 2026-03-29

### Added
- CLI entry point `tqs` with commands: `demo`, `benchmark`, `index`, `search`
- Pre-embedded dataset hub with Wikipedia and arxiv embeddings (auto-download + cache)
- Interactive comparison dashboard (TurboQuant vs FAISS side-by-side search)
- SIFT-1M benchmark support for large-scale evaluation
- Google Colab quickstart notebook
- "When to Use TurboQuant Search" guide in README
- "Stage 2 design choice" comparison table (sign-bit vs QJL trade-off)
- "Limitations & Honest Comparison" section with PQ scaling caveat
- GitHub Actions CI (pytest on push)

### Changed
- Gradio app rebuilt as comparison dashboard with search interface + live stats
- `requires-python` bumped to `>=3.9`
- Added `click`, `tqdm`, `requests` to core dependencies

## [0.1.0] - 2025-05-01

### Added
- Initial release
- TurboQuant compression (rotation + Lloyd-Max + sign-bit refinement)
- FAISS baselines (Flat, PQ, IVF-PQ)
- Benchmark runner with synthetic, SIFT-128, GloVe-100 datasets
- Gradio demo with benchmark, compression visualizer, memory calculator tabs
- 36 unit tests
