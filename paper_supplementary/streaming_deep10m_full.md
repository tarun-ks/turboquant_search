# Streaming at 10M scale: Deep-10M, full three-regime table

Referenced from Table 3 (Deep-10M, bit-matched only). Supplementary analysis; included here for completeness.

3 seeds (42, 123, 7777); mean ± 95% CI; paired-*t* on within-seed differences.
IVF-TQ at $b = 4$ + sign-bit refinement (512 bits/vec) across all regimes;
only IVF-PQ varies.

## IVF-TQ (single configuration, all regimes)

| Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|
| 512 | 87.39 ± 0.33 | 86.59 ± 0.17 | −0.80 ± 0.25 pp, $p = 0.005$ |

## Sub-matched (~0.75× IVF-TQ; $m = 48$, $b = 8$)

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 384 | 82.11 ± 0.15 | 78.87 ± 0.36 | −3.23 ± 0.49 pp, $p = 0.001$ |
| IVF-PQ retrain | 384 | 82.11 ± 0.15 | 78.94 ± 0.09 | −3.17 ± 0.15 pp, $p < 0.001$ |

IVF-TQ vs. PQ stale at 10M: **+7.72 ± 0.26 pp** ($p < 0.001$);
retrain − stale +0.06 ± 0.44 pp ($p = 0.595$).

## Bit-matched (~0.95× IVF-TQ; $m = 48$, $b = 10$) — also in main paper

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 480 | 87.21 ± 0.28 | 86.37 ± 0.21 | −0.84 ± 0.26 pp, $p = 0.005$ |
| IVF-PQ retrain | 480 | 87.21 ± 0.28 | 86.41 ± 0.23 | −0.80 ± 0.38 pp, $p = 0.012$ |

IVF-TQ vs. PQ stale at 10M: **+0.22 ± 0.05 pp** ($p = 0.003$);
retrain − stale +0.04 ± 0.43 pp ($p = 0.737$).

## Super-matched (~1.5× IVF-TQ; $m = 96$, $b = 8$)

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 768 | 91.30 ± 0.28 | 92.72 ± 0.51 | +1.42 ± 0.33 pp, $p = 0.003$ |
| IVF-PQ retrain | 768 | 91.30 ± 0.28 | 92.80 ± 0.17 | +1.50 ± 0.33 pp, $p = 0.003$ |

PQ stale exceeds IVF-TQ at 10M by **+6.13 ± 0.34 pp** ($p < 0.001$,
rate-distortion regime); retrain − stale +0.08 ± 0.42 pp ($p = 0.516$).

## Reproduction

`python experiments/streaming_multiseed.py --experiment deep10m{,_pqmatched,_pqhigh} --seeds 42 123 7777`
