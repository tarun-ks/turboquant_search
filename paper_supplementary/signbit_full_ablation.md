# Stage 2 sign-bit refinement: full Recall@10 and Recall@1 results

Referenced from main paper §3.2 (sign-bit refinement) and Appendix A
(flat-TQ ablation, summary Table 10). This file reports the full
9-cell ablation across three datasets and three bit budgets, at both
Recall@10 and Recall@1. Sign-bit refinement wins on every cell at both
metrics.

## Recall@10 (10K-scale)

| Dataset | Bits | No Stage 2 | QJL | Sign-bit (ours) |
|---|---|---|---|---|
| Synthetic | 2-bit | 54% | 58% | **72%** |
| Synthetic | 3-bit | 73% | 74% | **86%** |
| Synthetic | 4-bit | 85% | 87% | **92%** |
| SIFT-128  | 2-bit | 43% | 47% | **56%** |
| SIFT-128  | 3-bit | 59% | 62% | **73%** |
| SIFT-128  | 4-bit | 73% | 75% | **84%** |
| GloVe-100 | 2-bit | 56% | 59% | **72%** |
| GloVe-100 | 3-bit | 74% | 75% | **84%** |
| GloVe-100 | 4-bit | 84% | 85% | **92%** |

## Recall@1 (10K-scale)

Top-1 is more sensitive to ranking inversions, so the sign-bit refinement
advantage is even larger.

| Dataset | Bits | No Stage 2 | QJL | Sign-bit (ours) |
|---|---|---|---|---|
| Synthetic | 2-bit | 69% | 71% | **81%** |
| Synthetic | 3-bit | 83% | 84% | **87%** |
| Synthetic | 4-bit | 87% | 87% | **96%** |
| SIFT-128  | 2-bit | 30% | 33% | **44%** |
| SIFT-128  | 3-bit | 43% | 48% | **60%** |
| SIFT-128  | 4-bit | 63% | 66% | **74%** |
| GloVe-100 | 2-bit | 45% | 48% | **67%** |
| GloVe-100 | 3-bit | 68% | 70% | **83%** |
| GloVe-100 | 4-bit | 85% | 86% | **90%** |

## Notes

- All cells share Stage 1 (random rotation + Lloyd–Max scalar quantization);
  only the 1-bit Stage 2 differs.
- "No Stage 2" uses $b$ bits/coord; "QJL" and "Sign-bit" add 1 bit/coord
  (matched memory between the latter two).
- Sign-bit refinement uses the half-bin conditional mean
  $\hat{x}_j = \mathbb{E}[Z \mid Z \in \text{half-bin}(i, s_j)]$, with
  $s_j$ indicating which half of bin $i$ the value $\tilde{x}_j/\|\tilde{x}\|$
  falls in.
- The advantage shown here is **specific to the flat-TQ regime**.
  Under IVF the IVF-amplification effect collapses Stage 2 differences
  to noise (main paper Table 1, §3.3).
