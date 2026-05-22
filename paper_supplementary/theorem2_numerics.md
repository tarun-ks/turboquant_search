# Theorem 2 numerical evaluation at $d = 128$, $b = 4$, $\delta = 10^{-2}$

Referenced from main paper Appendix B (proofs of Theorem 1 and Theorem 2)
and §3.2 Tightness. This file holds the full term-by-term arithmetic
plugging into the bound of Theorem 2.

## Theorem 2 bound (restated)

$$|\langle q, v\rangle - \langle q, \hat{v}\rangle|
  \le D_b + R'_d + \sqrt{\frac{2 D_b (d \log(3d) + \log(2/\delta))}{d-1}} + \frac{2}{d}$$

uniformly over all $v \in S^{d-1}$ with probability $\ge 1 - \delta$ over
the random rotation $\Pi$.

## Term-by-term arithmetic at $d = 128$, $b = 4$, $\delta = 10^{-2}$

| Term | Value | Source |
|---|---|---|
| $D_b$ (Gaussian rate-distortion at 4 bits/dim) | $\approx 0.0104$ | Standard rate-distortion tables |
| $R'_d$ (bias-deviation term) | $\approx 0.97$ | $(2.355 + 1)\sqrt{2 \cdot 5.298 / 126}$ |
| Random term $\sqrt{2 D_b (d \log(3d) + \log(2/\delta))/(d-1)}$ | $\approx 0.354$ | $\sqrt{2 \cdot 0.0104 \cdot (128 \cdot 5.951 + 5.298)/127}$ |
| Lipschitz tail $2/d$ | $\approx 0.016$ | $2/128$ |
| **Total** | $\approx 1.35$ | sum |

## Comparison with Cauchy–Schwarz

Cauchy–Schwarz applied to Theorem 1's $\|v - \hat v\|_2$ bound at the same
parameters:

$$\sqrt{D_b} + R_d + \sqrt{8 \log(2/\delta)/(d-2)} \approx 0.102 + R_d + 0.580 \approx 0.68.$$

At $d = 128$ Cauchy–Schwarz is numerically tighter ($0.68 < 1.35$) because
Theorem 2's bias-deviation term $R'_d \approx 0.97$ (driven by $M + 1 \approx
3.36$ Lipschitz constant) dominates. The asymptotic $\sqrt{d/\log d}$
advantage of Theorem 2's random term over Cauchy–Schwarz's
$\sqrt{\log(1/\delta)/d}$ kicks in only at $d \gtrsim 10^3$.

## What Theorem 2 actually buys at $d = 128$

Theorem 2's structural advantages — uniformity over $S^{d-1}$ with one
fixed $\Pi$, and $(b, d, \delta)$-only dependence — are what carry the
streaming claim. The union bound in the proof is consumed once at index
initialisation over an $\epsilon$-net of the sphere whose log-size scales
with $d$, not over the database of $N$ arriving vectors. Adding the $N$-th
database vector consumes no additional budget. Cauchy–Schwarz on Theorem 1
is a per-pair (fixed-$v$) bound and admits no analogous uniform statement
at one fixed $\Pi$.

No learned-codebook PQ-family method (PQ, OPQ, ScaNN) admits an analogous
uniform, data-independent bound, because their reconstruction error
depends on the distance from $v$ to the nearest learned codebook centroid
— a function of the training sample.
