# Cascade search via Lloyd–Max bin ordinality

Referenced from main paper §3.2 (footnote). **Status: documented observation,
not a contribution.** This file records (i) the bit-importance asymmetry
between TQ and PQ that makes cascade meaningful in principle, and
(ii) a recall-preservation result for a two-pass cascade in our C++ reference.

The cascade yields a 2× speedup in our Python reference but ~1.01× in our
C++ reference (because the C++ kernel already reads scalar LUTs and a
smaller LUT does not help). The path to a real C++ speedup is FastScan-style
int8 LUTs with SIMD permute lookups ([andre2017fastscan]); we leave that
kernel work to follow-up. The tables below are retained so that interested
readers can verify the recall-preservation claim and reproduce the
bit-importance ablation, not because we claim cascade as an algorithmic
contribution.

## Bit-importance asymmetry: TQ vs. PQ

We measure the rank-relevance of each bit position by random-flip ablation:
corrupt $k\%$ of a chosen bit position across all stored codes, then re-measure
Recall@10. The drop quantifies how much that bit position contributes to
ranking quality. Result on SIFT-1M for IVF-RVQ-TQ ($b = 5 + 1$, 6 effective bits)
versus IVF-PQ ($m \in \{64, 128\}$).

| Method (baseline R@10) | Bit position | $k$=5% | $k$=10% | $k$=20% |
|---|---|---|---|---|
| IVF-RVQ-TQ b=5+1 (95.1%) | MSB of primary  | −63.3 pp | −77.5 pp | −89.4 pp |
|                          | LSB of primary  | −3.3 pp  | −5.2 pp  | −8.1 pp  |
|                          | Sign refinement | −1.1 pp  | −1.7 pp  | −3.3 pp  |
| IVF-PQ $m = 64$ (76.9%)  | MSB of code     | −56.6 pp | −64.7 pp | −71.2 pp |
|                          | Middle bit      | −56.7 pp | −65.2 pp | −71.3 pp |
|                          | LSB of code     | −57.1 pp | −65.4 pp | −71.1 pp |
|                          | Random byte     | −57.2 pp | −65.1 pp | −70.7 pp |
| IVF-PQ $m = 128$ (97.2%) | MSB of code     | −77.3 pp | −85.7 pp | −91.6 pp |
|                          | Middle bit      | −76.5 pp | −85.2 pp | −91.5 pp |
|                          | LSB of code     | −77.1 pp | −85.5 pp | −91.9 pp |
|                          | Random byte     | −76.9 pp | −85.2 pp | −91.8 pp |

- **TQ MSB:LSB ratio**: ~19:1.
- **PQ MSB:LSB ratio (both $m$)**: ~1:1 (max difference < 1 pp).

The ratio is the structural signature: TQ's primary bin index orders
centroids along the source distribution axis, so the MSB has direct
geometric meaning (sign of the rotated coordinate). PQ's bin index is
whatever ordering the $k$-means iterations happen to produce, so all bits
are interchangeable. We verified this at *two* PQ operating points to rule
out floor effects: at mid-recall ($m = 64$, baseline 76.9%) and at
high-recall ($m = 128$, baseline 97.2%). In both cases, MSB / Middle / LSB
/ Random differ by less than 1 pp at every corruption level — PQ codes
have no exploitable ordinal structure.

## Cascade algorithm: MSB-first filtering, full-precision re-rank

The bit-importance asymmetry suggests a two-pass search:

- **Pass 1 (coarse).** Score every candidate using only the top-$\beta$
  MSBs of the primary index. Reconstruct via a coarsened codebook with
  $2^\beta$ entries (the LSB-and-sign-refinement levels of each
  MSB-truncated bin are averaged into a single representative). Take top-$N$
  candidates.
- **Pass 2 (fine).** Re-rank the $N$ candidates from Pass 1 using the
  full primary + sign-refinement encoding.

Memory is unchanged (the same compressed encoding is used in both passes;
Pass 1 just masks LSBs at decode time). The hoped-for speedup comes from
(i) the smaller Pass-1 LUT (16 entries at $\beta = 4$ versus 64 at
$b = 5 + 1$), which would fit in NEON `vqtbl1q` and AVX2 `pshufb` for fast
SIMD gather; and (ii) Pass 2 amortising the slower full-precision decode
over only $N \ll$ database size candidates. Cascade requires that
lower-precision bin indices preserve the ordinal structure of
higher-precision indices — a property Lloyd–Max satisfies by construction
but $k$-means codebooks do not (bit-importance asymmetry above).

## Recall preservation in C++ across 16 conditions

Cascade in our production C++ reference (the same kernel used elsewhere
in the paper). Across 4 random-rotation seeds on each of Deep-1M and SIFT-1M,
plus an $n_p \in \{10, 20, 40, 80, 160\}$ sweep at seed 42, the cascade
preserves Recall@10 within ±0.02 pp of the baseline in every cell.

Configuration: $b = 5 + 1$, $\beta = 4$, $N = 100$.

### 4 seeds at $n_p = 40$

| Dataset | Seed | Baseline R@10 | Cascade R@10 | Δ | Speedup |
|---|---|---|---|---|---|
| Deep-1M | 42 | 94.99% | 94.98% | −0.01 pp | 1.00× |
| Deep-1M | 43 | 94.70% | 94.69% | −0.01 pp | 1.00× |
| Deep-1M | 44 | 94.61% | 94.61% |  0.00 pp | 1.00× |
| Deep-1M | 45 | 94.80% | 94.81% | +0.01 pp | 1.02× |
| SIFT-1M | 42 | 94.01% | 94.01% |  0.00 pp | 1.08× |
| SIFT-1M | 43 | 94.19% | 94.21% | +0.02 pp | 0.96× |
| SIFT-1M | 44 | 93.66% | 93.66% |  0.00 pp | 1.03× |
| SIFT-1M | 45 | 94.05% | 94.06% | +0.01 pp | 0.96× |

### $n_p$ sweep at seed 42

| Dataset | $n_p$ | Baseline R@10 | Cascade R@10 | Δ | Speedup |
|---|---|---|---|---|---|
| Deep-1M | 10  | 88.39% | 88.40% | +0.01 pp | 1.09× |
| Deep-1M | 20  | 93.06% | 93.06% |  0.00 pp | 0.98× |
| Deep-1M | 40  | 94.99% | 94.98% | −0.01 pp | 0.97× |
| Deep-1M | 80  | 95.66% | 95.66% |  0.00 pp | 0.98× |
| Deep-1M | 160 | 95.86% | 95.86% |  0.00 pp | 0.97× |
| SIFT-1M | 10  | 84.81% | 84.82% | +0.01 pp | 1.01× |
| SIFT-1M | 20  | 91.09% | 91.09% |  0.00 pp | 1.04× |
| SIFT-1M | 40  | 94.01% | 94.01% |  0.00 pp | 0.99× |
| SIFT-1M | 80  | 94.88% | 94.88% |  0.00 pp | 0.97× |
| SIFT-1M | 160 | 94.95% | 94.95% |  0.00 pp | 0.96× |

**Mean across 16 conditions**: Δ = +0.003 pp, speedup 1.01×.

## Why cascade does not yield a speedup in our C++ reference

The Python reference implementation showed a 2.0–2.3× speedup. The C++
reference shows essentially no speedup (1.01× mean) at preserved recall.
The discrepancy has a single cause: the Python baseline uses 2D NumPy fancy
indexing (`sub_centroids[primary, sub]`) per partition, which is
bandwidth-limited at the Python level and gets a real ~2× from a smaller
$\beta$-bit gather. The C++ baseline already implements the same scoring
path as a per-query ADC table (`table[d * n_entries + code[d]]` scalar
lookups), so reducing the LUT from 64 to 16 entries does not unlock anything
that wasn't already there: both lookups are cheap scalar indirect loads,
and both LUTs comfortably fit in L1 cache (3.2–12.3 KB at $d = 96$–128).

The actual lever for a C++ speedup is not a smaller LUT in floats but
**int8-quantised LUTs accessed through SIMD permute instructions** — the
FastScan technique ([andre2017fastscan]), where a 16-entry int8 LUT is
materialised in a NEON `vqtbl1q` register and 16 codes are looked up in
a single instruction. Implementing this in our kernel would re-introduce
the cascade speedup, but it requires careful int8 LUT scaling and the rest
of the kernel rewritten around the SIMD lane width; we leave it to future
work and do not claim a cascade speedup in C++.
