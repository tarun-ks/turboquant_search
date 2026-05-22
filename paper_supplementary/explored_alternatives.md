# Explored alternatives that did not work

Referenced from main paper §7 (Limitations). Two extensions investigated
during this work and found insufficient to be a main contribution.
Code for both is in the released artefact under `experiments/`.

## RH-IVF-TQ: random-hyperplane LSH partition + canonical centers

**Idea.** Replace the $k$-means coarse partition with $L$ random hyperplanes,
giving $2^L$ cells with hash-code-derived canonical centroids
$c(b) = \mathrm{normalize}(\sum_i b_i h_i)$ for hash bits $b \in \{-1, +1\}^L$.
Combined with TQ residual compression, this yields a *fully* data-independent
index: no $k$-means at any layer.

**Result on Deep-1M.** At $L \in \{8, 10, 12\}$ and $n_p \in \{10, 20, 40, 80\}$:
the best static recall configuration ($L = 8$, $n_p = 80$) achieves 79.2%
R@10 (rerank = 0) versus the IVF-TQ $k$-means baseline at 89.5% R@10 —
a ~10 pp deficit. Under the worst-case rotation shift described in
main paper §4.5, RH-IVF-TQ ends at 49.8% R@10 versus IVF-TQ frozen at
61.7% — dominated even in the regime where it should have an advantage.

**Diagnosis.** Random partitioning misses the cluster structure that real
ANN datasets exhibit. The recall floor it provides is mathematically
interesting (data-independent) but practically too low to compete with
even a frozen $k$-means partition under realistic shifts.

## FA-IVF-TQ: query-frequency-adaptive bit allocation

**Idea.** TQ's data-independent compression allows per-vector re-encoding
at any precision in $O(d)$ time, with no codebook retraining. We propose
maintaining per-vector hit counters and periodically re-encoding hot vectors
at higher bit precision (6 bits) and cold vectors at lower precision
(2 bits), exploiting Pareto skew in real query workloads.

**Result on Deep-1M with oracle hot set.** When the hot set is given by
ground-truth top-50 over popular queries, FA-IVF-TQ at average 2.02 bits/coord
(39.5 MB) matches uniform 4-bit (61.4 MB) on weighted recall (89.0% vs. 89.7%
at 80/20 popular/rare split). This is a 36% memory reduction at parity
recall — a significant Pareto improvement.

**Result on Deep-1M with realistic discovery.** When the hot set is identified
from a 5K-query warmup with the same skewed distribution (no ground truth),
FA-IVF-TQ collapses: weighted recall is 78.4% versus uniform 4-bit at 89.4%,
an 11.1 pp deficit. The hit-counter discovery mechanism, run on the warmup
index at 89% recall, fails to capture ~30% of the truly-hot vectors
(the ones the warmup index itself misses); demoting them to 2 bits collapses
popular-query recall when they reappear in held-out query top-K.

**Result at higher dim ($d = 768$).** On synthetic clustered 1M × 768-dim
data, the gap between uniform 2-bit (14.5% R@10) and uniform 6-bit (15.9% R@10)
collapses to ~1.4 pp — bit precision barely matters at high dim because
per-coordinate quantization errors average out. FA-IVF-TQ at 379.7 MB /
14.97% R@10 is Pareto-dominated by uniform 2-bit at 281.4 MB / 14.46%.
The premise of *spending more bits where queries hit* has no headroom in
this regime.

**Diagnosis.** Per-vector adaptive precision is a legitimate algorithmic
capability of TQ that no PQ-family method has, but it requires (i) steep
enough bit-precision sensitivity that hot/cold allocation matters
(favours low dim) and (ii) accurate hot-set discovery from query history
(favours high baseline recall). These two requirements are in tension:
low-dim regimes need it but warmup discovery is noisy; high-dim regimes
have accurate discovery but no bit-precision headroom. We document this
tension in case future work finds a regime that satisfies both — e.g.,
very-high-dim ($d \geq 4096$) embeddings with extreme query skew, or the
use of a higher-precision warmup index followed by quantization-aware
demotion.
