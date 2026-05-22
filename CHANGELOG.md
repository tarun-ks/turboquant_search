# Changelog

All notable changes to this project will be documented in this file.

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
