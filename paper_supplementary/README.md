# Paper Supplementary Materials

This directory contains supplementary materials for the PVLDB Vol 20 submission
**"IVF-TQ: Calibration-Free Streaming Vector Search via a Codebook-Free Residual Layer."**

Content here is referenced from the paper but moved out of the 12-page submission
to keep the main body + retained appendices (A, B Proofs, C Reproducibility)
within the PVLDB page limit. All claims, tables, and trajectories below are
reproducible from [`experiments/`](../experiments/) using the seeds listed in
each file.

## Files (organised by paper section)

### §3.2 IVF amplification

| File | Content |
|---|---|
| [`pq_ceiling_full.md`](pq_ceiling_full.md) | PQ recall ceiling on SIFT-1M across $m \in \{8, 16, 32, 64, 128\}$ |
| [`signbit_full_ablation.md`](signbit_full_ablation.md) | Full Recall@10 and Recall@1 Stage-2 ablation across 9 (dataset × bit) cells |
| [`cascade_observation.md`](cascade_observation.md) | Lloyd–Max bin-ordinality observation and two-pass cascade; bit-importance asymmetry and 16-condition recall-preservation evidence |

### §4.1 Streaming mechanism (1M controls)

| File | Content |
|---|---|
| [`streaming_controls_full.md`](streaming_controls_full.md) | SIFT-1M three-condition control (original/shuffled/mean-shift) |
| [`capacity_vs_bias_full.md`](capacity_vs_bias_full.md) | Capacity-vs-bias control on SIFT-1M (200K-initial / 200K-random / 1M-oracle) |

### §4.2 Streaming at 10M scale

| File | Content |
|---|---|
| [`streaming_deep10m_full.md`](streaming_deep10m_full.md) | Deep-10M, full three-regime (sub-/bit-/super-matched) 3-seed table |
| [`streaming_sift10m_full.md`](streaming_sift10m_full.md) | SIFT-10M, full three-regime 3-seed table |
| [`streaming_t2i10m_full.md`](streaming_t2i10m_full.md) | T2I-10M, full three-regime 3-seed table |
| [`streaming_deep10m_trajectory.md`](streaming_deep10m_trajectory.md) | Deep-10M single-seed per-batch trajectory |
| [`streaming_sift10m_trajectory.md`](streaming_sift10m_trajectory.md) | SIFT-10M single-seed per-batch trajectory |
| [`streaming_t2i10m_trajectory.md`](streaming_t2i10m_trajectory.md) | T2I-10M single-seed per-batch trajectory |

### §4.3 Encoder swap

| File | Content |
|---|---|
| [`encoder_swap_full.md`](encoder_swap_full.md) | Full per-batch encoder-swap trajectories on MS MARCO (3-seed; gentle L6→L12 and harsh L6→BGE) |

### §5 Million-Scale Comparison

| File | Content |
|---|---|
| [`million_scale_deep1m.md`](million_scale_deep1m.md) | Deep-1M block of the million-scale benchmark table |
| [`scann_baseline.md`](scann_baseline.md) | ScaNN baseline setup, anisotropic-AH parameters, and full $L_s$ sweep |

### §7 Limitations

| File | Content |
|---|---|
| [`explored_alternatives.md`](explored_alternatives.md) | Two extensions (RH-IVF-TQ, FA-IVF-TQ) explored and found insufficient as main contributions |

### Appendix B (Proofs)

| File | Content |
|---|---|
| [`theorem2_numerics.md`](theorem2_numerics.md) | Theorem 2 term-by-term numerical evaluation at $d{=}128$, $b{=}4$, $\delta{=}10^{-2}$ |

## Reproducibility

Scripts to regenerate the numbers in these files live in
[`../experiments/`](../experiments/). The full reproduction protocol is
documented in the [main README](../README.md) and in the Reproducibility
appendix of the paper.
