# SIFT-1M streaming controls: distribution-shift hypothesis test

Referenced from main paper §4.1 (Mechanism: capacity-bound, dataset-dependent).
Moved out of the paper to keep within the PVLDB 12-page limit. The
capacity-vs-bias control (Table 4 of the main paper) is the stronger of the
two §4.1 controls and remains in the paper; this file holds the
three-condition shift control.

## Setup

200K trained, 8 batches of 100K, $L = 500$, $n_p = 10$, 10K queries,
SIFT-1M. Three ingestion conditions test whether the streaming-recall
drop is explained by distribution shift.

## Three-condition control

| Condition | IVF-TQ (200K → 1M) | IVF-PQ (200K → 1M) | TQ $\Delta$ | PQ $\Delta$ |
|---|---|---|---|---|
| Original order              | 89.6 → 92.1% | 73.6 → 69.4% | +2.5 pp | **−4.2 pp** |
| Shuffled (i.i.d.)           | 89.3 → 92.0% | 73.2 → 69.4% | +2.7 pp | **−3.8 pp** |
| Mean-shift (0.05/batch)     | 89.6 → 91.4% | 73.6 → 70.8% | +1.8 pp | −2.8 pp |

## Reading

Under shuffled-i.i.d. ingestion (the streaming portion is statistically
identical to the training sample, so there is no distribution shift),
IVF-PQ at sub-matched memory still degrades **−3.8 pp** — comparable to the
−4.2 pp under original order. This rules out distribution shift as a
sufficient explanation for the streaming gap. (The mean-shift condition
shows a smaller −2.8 pp drop, which is also consistent with the bit-budget
mechanism since mean-shift on 0.05/batch is mild.)

IVF-TQ recall changes only modestly across the three conditions (+1.8 to
+2.7 pp); we attribute this to data-independent residual quality combined
with growing partition coverage.

We frame this as a *negative result* for the distribution-shift-only
hypothesis rather than as positive evidence for any specific alternative
mechanism. The capacity-vs-bias control in main paper Table 4 narrows the
alternative space further, and the 9-cell streaming matrix at 10M scale
(main paper Tables 5–7) establishes that retraining the codebook is not the
lever that closes the streaming gap.
