# PQ recall ceiling (SIFT-1M)

Referenced from §3.2 (IVF amplification). Supplementary analysis; included here for completeness.
for any fixed PQ configuration, IVF-PQ exhibits a per-configuration recall
ceiling that scales with the bit budget; IVF-TQ's ceiling is set by $(b, d)$
alone via the rate-distortion floor $\sqrt{D_b}$.

## PQ recall ceiling on SIFT-1M

$n_p = 160$, 10K queries. Ceiling scales with bit budget $m$.
IVF-TQ has no per-configuration ceiling.

| PQ $m$ | Bits/dim | Ceiling R@10 | Memory |
|---|---|---|---|
| 8   | 0.5 | 14.0% | 8 MB |
| 16  | 1.0 | 28.1% | 16 MB |
| 32  | 2.0 | 46.3% | 31 MB |
| 64  | 4.0 | 73.2% | 62 MB |
| 128 | 8.0 | 92.9% | 123 MB |
| **IVF-TQ 4-bit (5.0 bits/dim)** | | **87.5%** | **81 MB** |

## Reading

The ceilings climb monotonically with $m$ (bit budget) but plateau at each
$m$: increasing $n_p$ beyond 160 yields essentially no further recall.
IVF-PQ's quantization error — not partition coverage — is the binding
constraint at each $m$. The same capacity issue surfaces in the streaming
results (main paper §4): at sub-matched memory ($m = 48$ on Deep, $m = 64$
on SIFT), the codebook is too small for the eventual 10M database, and
retraining cannot recover the gap (Tables 5, 6, 7 of the main paper).

IVF-TQ at 4-bit (5 effective bits/coord) reaches 87.5% R@10 at $n_p = 20$
(81 MB), within $\sim 1$ pp of FAISS PQ $m = 64$ (4 bits/dim, 62 MB)
on the static-recall axis. The IVF-TQ ceiling is set by the
rate-distortion floor $\sqrt{D_b}$ on residual reconstruction
(Theorem 1 of the main paper); higher error floor implies lower
recall ceiling, and the ceiling is $(b, d)$-only.
