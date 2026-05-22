# Streaming at 10M scale: T2I-10M, full three-regime table

Referenced from main paper Table 5 (T2I-10M, bit-matched only). The
sub-matched and super-matched blocks below were moved out of the main
paper to keep within the PVLDB 12-page limit.

3 seeds (42, 123, 7777); mean ± 95% CI. IVF-TQ at $b = 4$ + sign-bit
refinement (1032 bits/vec) across all regimes; only IVF-PQ varies.
$m_{\text{PQ}}$ is constrained by $d \% m_{\text{PQ}} = 0$, so the standard
$m = 48$ regime used for Deep-10M is not available at $d = 200$; the
closest sub-matched configuration is $m = 100$, $b = 8$.

Memory ratios computed against IVF-TQ's 1032 bits/vec: sub-matched 800 bits
(~0.78×); bit-matched 1000 bits (~0.97×); super-matched 1600 bits (~1.55×).

## IVF-TQ (single configuration, all regimes)

| Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|
| 1032 | 81.64 ± 0.33 | 80.88 ± 0.10 | −0.76 ± 0.41 pp, $p = 0.015$ |

## Sub-matched (~0.78× IVF-TQ; $m = 100$, $b = 8$)

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 800 | 76.71 ± 0.18 | 73.47 ± 0.14 | −3.24 ± 0.28 pp, $p < 0.001$ |
| IVF-PQ retrain | 800 | 76.71 ± 0.18 | 73.37 ± 0.42 | −3.34 ± 0.49 pp, $p = 0.002$ |

IVF-TQ vs. PQ stale at 10M: **+7.41 ± 0.12 pp** ($p < 0.001$);
retrain − stale −0.10 ± 0.30 pp ($p = 0.291$).

## Bit-matched (~0.97× IVF-TQ; $m = 100$, $b = 10$) — also in main paper

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 1000 | 81.48 ± 0.24 | 80.59 ± 0.22 | −0.89 ± 0.32 pp, $p = 0.007$ |
| IVF-PQ retrain | 1000 | 81.48 ± 0.24 | 80.50 ± 0.06 | −0.98 ± 0.23 pp, $p = 0.003$ |

IVF-TQ vs. PQ stale at 10M: **+0.29 ± 0.24 pp** ($p = 0.018$);
retrain − stale −0.09 ± 0.28 pp ($p = 0.317$).

## Super-matched (~1.55× IVF-TQ; $m = 200$, $b = 8$)

| Index | Bits/vec | R@10 (1M) | R@10 (10M) | Change Δ |
|---|---|---|---|---|
| IVF-PQ stale   | 1600 | 84.94 ± 0.19 | 86.01 ± 0.49 | +1.07 ± 0.45 pp, $p = 0.010$ |
| IVF-PQ retrain | 1600 | 84.94 ± 0.19 | 85.99 ± 0.41 | +1.05 ± 0.34 pp, $p = 0.006$ |

PQ stale exceeds IVF-TQ at 10M by **+5.12 ± 0.45 pp** ($p < 0.001$,
rate-distortion); retrain − stale −0.01 ± 0.12 pp ($p = 0.682$).

## Reproduction

`python experiments/streaming_multiseed.py --experiment t2i10m{,_pqmatched,_pqhigh} --seeds 42 123 7777`
