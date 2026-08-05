# Streaming at 10M scale: SIFT-10M, full three-regime table

Referenced from Table 4 (SIFT-10M, bit-matched only). Supplementary analysis; included here for completeness.

3 seeds (42, 123, 7777); mean ± 95% CI. IVF-TQ at $b = 4$ + sign-bit
refinement (672 bits/vec) across all regimes; only IVF-PQ varies.

## IVF-TQ (single configuration, all regimes)

| Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|
| 672 | 83.91 ± 0.08 | 84.47 ± 0.28 | +0.56 ± 0.10 pp, $p = 0.007$ |

## Sub-matched (~0.75× IVF-TQ; $m = 64$, $b = 8$)

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 512 | 72.42 ± 0.41 | 66.61 ± 0.70 | −5.80 ± 0.55 pp, $p < 0.001$ |
| IVF-PQ retrain | 512 | 72.42 ± 0.41 | 66.78 ± 0.25 | −5.64 ± 0.66 pp, $p < 0.001$ |

IVF-TQ vs. PQ stale at 10M: **+17.86 ± 0.47 pp** ($p < 0.001$);
retrain − stale +0.17 ± 0.50 pp ($p = 0.289$).

## Bit-matched (~0.95× IVF-TQ; $m = 64$, $b = 10$) — also in main paper

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 640 | 80.00 ± 0.44 | 77.69 ± 0.04 | −2.31 ± 0.42 pp, $p = 0.002$ |
| IVF-PQ retrain | 640 | 80.00 ± 0.44 | 77.61 ± 0.12 | −2.40 ± 0.38 pp, $p = 0.001$ |

IVF-TQ vs. PQ stale at 10M: **+6.79 ± 0.25 pp** ($p < 0.001$);
retrain − stale −0.08 ± 0.07 pp ($p = 0.040$, negligible magnitude and
opposite in sign to a recovery effect).

## Super-matched (~1.5× IVF-TQ; $m = 128$, $b = 8$)

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 1024 | 85.42 ± 0.16 | 86.50 ± 0.31 | +1.09 ± 0.40 pp, $p = 0.001$ |
| IVF-PQ retrain | 1024 | 85.42 ± 0.16 | 86.60 ± 0.34 | +1.18 ± 0.34 pp, $p = 0.004$ |

PQ stale exceeds IVF-TQ at 10M by **+2.03 ± 0.10 pp** ($p < 0.001$,
rate-distortion); retrain − stale +0.09 ± 0.31 pp ($p = 0.317$).

## Reproduction

`python experiments/streaming_multiseed.py --experiment sift10m{,_pqmatched,_pqhigh} --seeds 42 123 7777`
